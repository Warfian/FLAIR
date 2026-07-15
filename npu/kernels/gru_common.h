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
//  * Nonlinearities are built from the EXP LUT (getExpBf16, truncate policy,
//    never NaN), NOT getTanhBf16 (which had a deterministic NaN on an
//    interior input).
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

// sigmoid(x) = 1 / (1 + exp(-x)).
//
// exp comes from getExpBf16 (a real vector LUT). The RECIPROCAL is where this
// kernel used to lose all its accuracy: getInvBf16 is a bit-manipulation
// reciprocal (the classic magic-constant trick), accurate to only ~5-8% without
// refinement. Because sigmoid's relative error IS the reciprocal's relative
// error, that one function plausibly accounts for the entire ~8-9% nonlinearity
// error -- which inflates the normal-p99 detection threshold ~6x and costs ~4 F1
// points on hardware (NPU 0.894 vs PyTorch 0.936, missing 743 attacks instead of
// 68 at the same 1% FPR). See npu/precision_ablation.py.
//
// Fix: keep getInvBf16 only as a SEED, then refine with two Newton-Raphson
// steps:  r <- r * (2 - denom*r).  The relative error SQUARES each step
// (~5% -> ~0.25% -> ~6e-6), so the reciprocal stops being the bottleneck and
// sigmoid inherits the exp LUT's accuracy instead.
//
// This deliberately introduces NO new primitives. getExpBf16 and the per-lane
// getInvBf16 loop are exactly as in the kernel that already works, and Newton
// needs only vector mul/sub -- the same ops matvec_bias and the gate loop
// already rely on. In particular: NO aie::max/min, and NO scalar fp32
// arithmetic. Earlier attempts at this fix used those and produced divergence
// (scalar fp32) or a badly inaccurate tanh (aie::max/min) on hardware.
//
// Every aie::mul/add/sub result is materialized into a named vector -- they
// return an aie::accum, NOT a vector, and chaining them corrupts every lane.
//
// Safety: denom = 1 + exp(-x) >= 1 always, so Newton can never divide by zero,
// and it converges from any seed within (0, 2/denom) -- which a reciprocal
// approximation of any sane quality satisfies.
inline aie::vector<bfloat16, 16>
sigmoid16(const aie::vector<bfloat16, 16> &x) {
  aie::vector<bfloat16, 16> zero = aie::broadcast<bfloat16, 16>(0.0f);
  aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  aie::vector<bfloat16, 16> two = aie::broadcast<bfloat16, 16>(2.0f);
  aie::vector<bfloat16, 16> neg_x = aie::sub(zero, x);              // -x
  aie::vector<bfloat16, 16> e = to_v16bfloat16(getExpBf16(neg_x));  // exp(-x)
  aie::vector<bfloat16, 16> denom = aie::add(one, e);               // 1 + exp(-x)

  // Coarse seed, per lane -- unchanged from the working kernel.
  alignas(aie::vector_decl_align) bfloat16 denom_arr[16];
  alignas(aie::vector_decl_align) bfloat16 seed_arr[16];
  aie::store_v(denom_arr, denom);
  for (int j = 0; j < 16; j++)
    seed_arr[j] = getInvBf16((float)denom_arr[j]); // ~1/denom, only ~5-8% exact
  aie::vector<bfloat16, 16> r0 = aie::load_v<16>(seed_arr);

  // Newton step 1: r1 = r0 * (2 - denom*r0)
  aie::vector<bfloat16, 16> dr0 = aie::mul(denom, r0);
  aie::vector<bfloat16, 16> c0 = aie::sub(two, dr0);
  aie::vector<bfloat16, 16> r1 = aie::mul(r0, c0);

  // Newton step 2: r2 = r1 * (2 - denom*r1)  (the relative error squares again)
  aie::vector<bfloat16, 16> dr1 = aie::mul(denom, r1);
  aie::vector<bfloat16, 16> c1 = aie::sub(two, dr1);
  aie::vector<bfloat16, 16> r2 = aie::mul(r1, c1);

  return r2;
}

// tanh(x) = 2*sigmoid(2x) - 1.
inline aie::vector<bfloat16, 16>
tanh16(const aie::vector<bfloat16, 16> &x) {
  aie::vector<bfloat16, 16> two = aie::broadcast<bfloat16, 16>(2.0f);
  aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  aie::vector<bfloat16, 16> two_x = aie::mul(x, two);
  aie::vector<bfloat16, 16> s = sigmoid16(two_x);
  aie::vector<bfloat16, 16> two_s = aie::mul(two, s);
  return aie::sub(two_s, one);
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

  // No h_prev copy: gh is computed from the FULL old h below, and the gate
  // combine is elementwise, so each 16-lane block can simply load h[i..i+15]
  // immediately before it overwrites them. Dropping that array frees 128B of the
  // ~1KB core stack, giving Newton's extra vector registers room to live.
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

  // No h_prev copy -- see gru_step. Frees 128B of the ~1KB core stack.
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
