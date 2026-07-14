//===- gru_common.h ---------------------------------------*- C++ -*-===//
//
// Shared AIE kernel building blocks for the FLAIR GRU designs (encoder,
// decoder, full forward). Factored out of the validated single-cell kernel
// (gru_cell.cc) so the multi-timestep designs reuse the exact same,
// hardware-verified math.
//
// Include requirements (provided by the IRON driver's include_dirs + the
// source_string wrapper that compiles lut_based_ops.cpp into the TU):
//   aie_kernel_utils.h, aie_api/aie.hpp, lut_based_ops.h
// The including .cc must pull those in before this header.
//
// Key learned constraints (see the FLAIR NPU memory / commit history):
//  * aie::mul/add/sub of two bf16 vectors return an aie::accum, not a vector;
//    materialize each into an explicit vector before feeding the next op.
//  * Nonlinearities are a minimax polynomial tanh on the VECTOR unit, with
//    sigmoid = 0.5*(1+tanh(x/2)) -- see the block below. Do NOT use
//    getTanhBf16 (deterministic NaN on an interior input). Do NOT evaluate them
//    in scalar fp32: two attempts did, and both DIVERGED on hardware.
//  * The scalar fp32 path on this core is not trustworthy. Note matvec_bias
//    always takes its VECTORIZED branch here (cols 48 and 64 are both %16==0),
//    so its scalar fallback never actually runs.
//  * aie::load_v<16>/store_v need 32-byte-aligned pointers; copy any buffer
//    reached at a non-vector-aligned offset into an aligned local first.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef FLAIR_GRU_COMMON_H
#define FLAIR_GRU_COMMON_H

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

static_assert(HIDDEN_DIM % 16 == 0,
             "HIDDEN_DIM must be a multiple of 16 (16-lane bf16 vector ops)");

namespace flair {

// ---------------------------------------------------------------------------
// Nonlinearities: minimax polynomial tanh evaluated ENTIRELY ON THE VECTOR
// UNIT, with sigmoid derived as 0.5*(1 + tanh(x/2)).
//
// WHY NOT THE EXP LUT: the old getExpBf16 + getInvBf16 sigmoid is NaN-free but
// only ~8-9% accurate, and that error is the whole NPU accuracy loss -- it
// inflates the normal-p99 detection threshold ~6x and costs ~4 F1 points
// (NPU 0.894 vs PyTorch 0.936). bf16 rounding itself costs nothing. See
// npu/precision_ablation.py, which reproduces this without hardware.
//
// WHY VECTOR, NOT SCALAR -- READ THIS BEFORE "SIMPLIFYING":
// Two earlier attempts evaluated this same polynomial (and a Pade[7/6]
// rational) in SCALAR fp32. Both DIVERGED on hardware: latents reached ~4e8,
// though a GRU hidden state is mathematically confined to [-1,1]. The scalar
// fp32 path on this core cannot be trusted -- and note the rest of this file
// never depends on it either: matvec_bias always takes its VECTORIZED branch
// (both cols=48 and cols=64 are multiples of 16), so its scalar fallback never
// actually runs. Everything below therefore uses aie:: vector ops only.
//
// EVERY aie::mul/add/sub RESULT IS MATERIALIZED into a named vector. Those
// return an aie::accum, NOT a vector; chaining them -- e.g.
// aie::mul(aie::add(a,b), c) -- compiles fine but silently yields WRONG VALUES
// IN EVERY LANE (the "all 64 latent dims mismatch" symptom). Do not collapse
// these statements.
//
// Simulated with bf16 rounding after every op (exactly what the vector unit
// does): |tanh| <= 1 and sigmoid in [0,1] (guaranteed by the output clamp), and
// end-to-end this restores the pipeline to inflation 1.02x / F1 0.938
// (PyTorch: 0.937).
// ---------------------------------------------------------------------------

// Odd minimax polynomial for tanh on [-3, 3], evaluated by Horner in x^2:
//   tanh(x) ~ x * (t1 + t3 x^2 + t5 x^4 + t7 x^6 + t9 x^8 + t11 x^10)
static constexpr float kT1 = 9.8983037472e-01f;
static constexpr float kT3 = -2.8898122907e-01f;
static constexpr float kT5 = 7.3146112263e-02f;
static constexpr float kT7 = -1.1429719627e-02f;
static constexpr float kT9 = 9.4607059145e-04f;
static constexpr float kT11 = -3.1460636819e-05f;

// Clamp x into [lo, hi]. Vector-only (aie::max/min return vectors, not accums).
inline aie::vector<bfloat16, 16>
clamp16(const aie::vector<bfloat16, 16> &x, float lo, float hi) {
  aie::vector<bfloat16, 16> vlo = aie::broadcast<bfloat16, 16>(lo);
  aie::vector<bfloat16, 16> vhi = aie::broadcast<bfloat16, 16>(hi);
  aie::vector<bfloat16, 16> t = aie::max(x, vlo);
  aie::vector<bfloat16, 16> r = aie::min(t, vhi);
  return r;
}

inline aie::vector<bfloat16, 16>
tanh16(const aie::vector<bfloat16, 16> &x) {
  // INPUT CLAMP IS MANDATORY: a polynomial diverges outside its fit interval,
  // so an unclamped evaluation returns |tanh| >> 1 on large gate
  // pre-activations and blows up the recurrence. tanh(3)=0.995, so this costs
  // < 5e-3.
  aie::vector<bfloat16, 16> v = clamp16(x, -3.0f, 3.0f);
  aie::vector<bfloat16, 16> v2 = aie::mul(v, v); // materialize

  // Horner in v2. Each coefficient is broadcast AT ITS POINT OF USE so only one
  // is live at a time -- keeping all five live at once raises vector-register
  // pressure enough to spill onto the ~1KB core stack, which is what made the
  // decoder (the tighter of the two, ~1KB before spills) produce NaN while the
  // encoder was already clean. Every aie::mul/add is still materialized into a
  // named vector; broadcast returns a vector, so this is not accum chaining.
  aie::vector<bfloat16, 16> r = aie::broadcast<bfloat16, 16>(kT11);
  aie::vector<bfloat16, 16> t;

  t = aie::mul(v2, r);
  r = aie::add(t, aie::broadcast<bfloat16, 16>(kT9));
  t = aie::mul(v2, r);
  r = aie::add(t, aie::broadcast<bfloat16, 16>(kT7));
  t = aie::mul(v2, r);
  r = aie::add(t, aie::broadcast<bfloat16, 16>(kT5));
  t = aie::mul(v2, r);
  r = aie::add(t, aie::broadcast<bfloat16, 16>(kT3));
  t = aie::mul(v2, r);
  r = aie::add(t, aie::broadcast<bfloat16, 16>(kT1));

  aie::vector<bfloat16, 16> out = aie::mul(v, r); // materialize
  // OUTPUT CLAMP: bf16 rounding through the Horner chain can overshoot to
  // ~1.03, and the GRU needs |tanh| <= 1 to keep h inside [-1,1].
  return clamp16(out, -1.0f, 1.0f);
}

// sigmoid(x) = 0.5 * (1 + tanh(x/2)). Bounded in [0,1] by the tanh clamp above,
// so there is exactly ONE approximation in the kernel to validate.
inline aie::vector<bfloat16, 16>
sigmoid16(const aie::vector<bfloat16, 16> &x) {
  aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  aie::vector<bfloat16, 16> hx = aie::mul(x, half); // materialize
  aie::vector<bfloat16, 16> t = tanh16(hx);
  aie::vector<bfloat16, 16> s = aie::add(one, t);    // materialize
  aie::vector<bfloat16, 16> out = aie::mul(s, half); // materialize
  return out;
}

// out[row] = sum_i(w[row*cols + i] * in[i]) + bias[row]. bf16 in/out, fp32
// accumulation. `bias` may be nullptr.
//
// Two paths:
//  * cols a multiple of 16  -> VECTORIZED per-row dot product (16-lane bf16
//    MAC into an accfloat accumulator + reduce_add). This path requires the
//    weight rows AND `in` to be 32-byte (aie::vector_decl_align) aligned:
//    a row starts at w + row*cols, so cols%16==0 keeps every row aligned iff
//    `w` and `in` are aligned. Callers must guarantee that (the encoder's
//    w_hh @ h qualifies: w_hh rows are 128-byte-strided and h is aligned).
//  * otherwise -> scalar fallback (handles e.g. the encoder's w_ih, cols=45,
//    until the weights are padded to a multiple of 16).
inline void matvec_bias(const bfloat16 *restrict w, const bfloat16 *restrict in,
                        const bfloat16 *restrict bias, bfloat16 *restrict out,
                        int rows, int cols) {
  if ((cols & 15) == 0) {
    for (int row = 0; row < rows; row++) {
      const bfloat16 *restrict w_row = w + row * cols;
      aie::accum<accfloat, 16> acc = aie::mul(aie::load_v<16>(w_row),
                                              aie::load_v<16>(in));
      for (int i = 16; i < cols; i += 16) {
        aie::vector<bfloat16, 16> wv = aie::load_v<16>(w_row + i);
        aie::vector<bfloat16, 16> iv = aie::load_v<16>(in + i);
        acc = aie::mac(acc, wv, iv);
      }
      float dot = aie::reduce_add(acc.to_vector<float>());
      out[row] = (bfloat16)(dot + (bias ? (float)bias[row] : 0.0f));
    }
  } else {
    for (int row = 0; row < rows; row++) {
      float acc = bias ? (float)bias[row] : 0.0f;
      const bfloat16 *restrict w_row = w + row * cols;
      for (int i = 0; i < cols; i++)
        acc += (float)w_row[i] * (float)in[i];
      out[row] = (bfloat16)acc;
    }
  }
}

// One GRU timestep, updating h in place (PyTorch nn.GRUCell math):
//   gi = w_ih @ x_in + b_ih ; gh = w_hh @ h + b_hh
//   r = sigmoid(gi_r+gh_r) ; z = sigmoid(gi_z+gh_z)
//   n = tanh(gi_n + r*gh_n) ; h <- (1-z)*n + z*h
// Gate order [reset, update, new] (PyTorch weight_ih/hh layout).
//
// `h` MUST be 32-byte (aie::vector_decl_align) aligned and HIDDEN_DIM long;
// the combine loop vector-loads/stores it. `x_in` is read scalar-only, so it
// may sit at any offset (e.g. a per-timestep slice of a window buffer).
// `input_dim` is the length of x_in (encoder: 45, decoder: HIDDEN_DIM).
inline void gru_step(const bfloat16 *restrict x_in, bfloat16 *restrict h,
                     const bfloat16 *restrict w_ih,
                     const bfloat16 *restrict w_hh,
                     const bfloat16 *restrict b_ih,
                     const bfloat16 *restrict b_hh, int input_dim) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  alignas(aie::vector_decl_align) bfloat16 gi[H3];
  alignas(aie::vector_decl_align) bfloat16 gh[H3];

  matvec_bias(w_ih, x_in, b_ih, gi, H3, input_dim);
  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses the FULL old h -- must run
                                         // before the combine loop touches h

  const bfloat16 *gi_r = gi, *gi_z = gi + H, *gi_n = gi + 2 * H;
  const bfloat16 *gh_r = gh, *gh_z = gh + H, *gh_n = gh + 2 * H;

  AIE_LOOP_MIN_ITERATION_COUNT(H / 16)
  for (int i = 0; i < H; i += 16) {
    aie::vector<bfloat16, 16> vgi_r = aie::load_v<16>(gi_r + i);
    aie::vector<bfloat16, 16> vgh_r = aie::load_v<16>(gh_r + i);
    aie::vector<bfloat16, 16> pre_r = aie::add(vgi_r, vgh_r);
    aie::vector<bfloat16, 16> r = sigmoid16(pre_r);

    aie::vector<bfloat16, 16> vgi_z = aie::load_v<16>(gi_z + i);
    aie::vector<bfloat16, 16> vgh_z = aie::load_v<16>(gh_z + i);
    aie::vector<bfloat16, 16> pre_z = aie::add(vgi_z, vgh_z);
    aie::vector<bfloat16, 16> z = sigmoid16(pre_z);

    aie::vector<bfloat16, 16> vgi_n = aie::load_v<16>(gi_n + i);
    aie::vector<bfloat16, 16> vgh_n = aie::load_v<16>(gh_n + i);
    aie::vector<bfloat16, 16> r_gh_n = aie::mul(r, vgh_n);
    aie::vector<bfloat16, 16> n_pre = aie::add(vgi_n, r_gh_n);
    aie::vector<bfloat16, 16> n = tanh16(n_pre);

    // Old h[i..i+15], loaded BEFORE this block's store below. No h_prev copy is
    // needed: gh was already computed from the full old h above, and the
    // combine is elementwise, so each 16-lane block only needs its own old h.
    // (Dropping that array frees 128B of the ~1KB core stack.)
    aie::vector<bfloat16, 16> vh_prev = aie::load_v<16>(h + i);
    aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
    aie::vector<bfloat16, 16> one_minus_z = aie::sub(one, z);
    aie::vector<bfloat16, 16> term1 = aie::mul(one_minus_z, n);
    aie::vector<bfloat16, 16> term2 = aie::mul(z, vh_prev);
    aie::vector<bfloat16, 16> h_out = aie::add(term1, term2);

    aie::store_v(h + i, h_out); // overwrite h with the new hidden state
  }
}

// Variant of gru_step for callers whose x_in is CONSTANT across the whole
// timestep loop (e.g. the decoder: x_t = h0_vec on every step, since it has
// no autoregressive/categorical feedback into the recurrence -- see
// decoder.py's "repeated input" design). gi = w_ih @ x_in + b_ih is then
// invariant too, so the caller computes it ONCE (via matvec_bias) before the
// loop and passes it here each timestep, instead of gru_step recomputing it
// from x_in on every call. Only gh = w_hh @ h + b_hh (which genuinely
// changes as h evolves) is recomputed. Same gate-combine math as gru_step.
//
// `gi` MUST be 32-byte aligned (vector-loaded) and H3=3*HIDDEN_DIM long.
// `h` MUST be 32-byte aligned and HIDDEN_DIM long, same as gru_step.
inline void gru_step_with_gi(const bfloat16 *restrict gi, bfloat16 *restrict h,
                             const bfloat16 *restrict w_hh,
                             const bfloat16 *restrict b_hh) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  // No h_prev array (see gru_step): each 16-lane block loads its own old h just
  // before overwriting it. This is the tightest kernel on the ~1KB core stack --
  // gru_decoder.cc's caller already holds gi[192]+h[64] (512B) live across the
  // timestep loop -- so the 128B saved here matters.
  alignas(aie::vector_decl_align) bfloat16 gh[H3];

  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses the FULL old h -- must run
                                         // before the combine loop touches h

  const bfloat16 *gi_r = gi, *gi_z = gi + H, *gi_n = gi + 2 * H;
  const bfloat16 *gh_r = gh, *gh_z = gh + H, *gh_n = gh + 2 * H;

  AIE_LOOP_MIN_ITERATION_COUNT(H / 16)
  for (int i = 0; i < H; i += 16) {
    aie::vector<bfloat16, 16> vgi_r = aie::load_v<16>(gi_r + i);
    aie::vector<bfloat16, 16> vgh_r = aie::load_v<16>(gh_r + i);
    aie::vector<bfloat16, 16> pre_r = aie::add(vgi_r, vgh_r);
    aie::vector<bfloat16, 16> r = sigmoid16(pre_r);

    aie::vector<bfloat16, 16> vgi_z = aie::load_v<16>(gi_z + i);
    aie::vector<bfloat16, 16> vgh_z = aie::load_v<16>(gh_z + i);
    aie::vector<bfloat16, 16> pre_z = aie::add(vgi_z, vgh_z);
    aie::vector<bfloat16, 16> z = sigmoid16(pre_z);

    aie::vector<bfloat16, 16> vgi_n = aie::load_v<16>(gi_n + i);
    aie::vector<bfloat16, 16> vgh_n = aie::load_v<16>(gh_n + i);
    aie::vector<bfloat16, 16> r_gh_n = aie::mul(r, vgh_n);
    aie::vector<bfloat16, 16> n_pre = aie::add(vgi_n, r_gh_n);
    aie::vector<bfloat16, 16> n = tanh16(n_pre);

    // Old h[i..i+15], loaded BEFORE this block's store below (no h_prev copy).
    aie::vector<bfloat16, 16> vh_prev = aie::load_v<16>(h + i);
    aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
    aie::vector<bfloat16, 16> one_minus_z = aie::sub(one, z);
    aie::vector<bfloat16, 16> term1 = aie::mul(one_minus_z, n);
    aie::vector<bfloat16, 16> term2 = aie::mul(z, vh_prev);
    aie::vector<bfloat16, 16> h_out = aie::add(term1, term2);

    aie::store_v(h + i, h_out); // overwrite h with the new hidden state
  }
}

} // namespace flair

#endif // FLAIR_GRU_COMMON_H
