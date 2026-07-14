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
//    Chaining them (e.g. aie::mul(aie::add(a,b), c)) silently yields wrong
//    values in EVERY lane -- the tell is an all-dimensions mismatch.
//  * Nonlinearities are a Pade[7/6] rational tanh evaluated in scalar fp32,
//    with sigmoid derived as 0.5*(1+tanh(x/2)) -- see the block above them.
//    Do NOT use getTanhBf16 (deterministic NaN on an interior input). The old
//    exp-LUT sigmoid (getExpBf16 + raw getInvBf16) was NaN-free but only ~8-9%
//    accurate, which was the entire NPU accuracy loss; it is gone.
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
// Nonlinearities: division-free polynomial tanh, sigmoid derived from it.
//
// WHY: the old exp-LUT path (getExpBf16 + a raw getInvBf16 reciprocal) was only
// ~8-9% accurate, and that was the ENTIRE NPU accuracy loss -- it inflated the
// normal-p99 detection threshold 6x and cost ~4 F1 points (NPU 0.894 vs PyTorch
// 0.936). bf16 rounding itself costs NOTHING (1.00x inflation, F1 identical to
// PyTorch), so the nonlinearity was the whole error budget. See
// npu/precision_ablation.py, which reproduces all of this without hardware.
//
// HOW (and what it deliberately avoids -- do not "simplify" these away):
//
//  1. THE CLAMP IS MANDATORY. A polynomial diverges outside its fit interval,
//     so an unclamped evaluation returns |tanh| >> 1 on large gate
//     pre-activations and destabilises the recurrence. Same trap as the earlier
//     Pade form, whose num/den -> 0.0357*x as |x| -> inf.
//
//  2. NO DIVISION, NO getInvBf16, NO LUT -- multiplies and adds only. An
//     equally accurate Pade[7/6] rational was tried first, but it needs a
//     reciprocal, and getInvBf16 + Newton was the one component never validated
//     on this hardware; the encoder NaN'd with it. Removing the division
//     removes that unknown. (If this STILL NaNs, the remaining suspect is
//     scalar fp32 arithmetic on the AIE core itself -- note the old kernel
//     barely used any: matvec_bias takes its VECTORIZED path for both cols=48
//     and cols=64, so its scalar fallback never runs. The fix would then be to
//     evaluate this polynomial with aie:: vector ops instead.)
//
//  3. SCALAR, *not* chained aie:: vector ops. aie::mul/add/sub on bf16 vectors
//     return an aie::accum, NOT a vector; chaining them (e.g.
//     aie::mul(aie::add(a,b),c)) silently yields WRONG VALUES IN EVERY LANE --
//     the "all 64 latent dims mismatch" symptom. gru_gate_combine is therefore
//     a plain scalar loop, which sidesteps that trap, needs no store_v/load_v
//     scratch, and keeps bf16 rounding out of the polynomial. matvec_bias stays
//     vectorized, so almost none of the FLOPs are lost.
//
// STACK: only a handful of fp32 temporaries, no scratch arrays. gi/gh are also
// static (see gru_step), so the ~1KB core stack has ample headroom.
// ---------------------------------------------------------------------------

// Minimax-fitted ODD polynomial for tanh, degree 11, valid on [-3, 3]:
//   tanh(x) ~ x * (t1 + t3 x^2 + t5 x^4 + t7 x^6 + t9 x^8 + t11 x^10)
// evaluated by Horner in x^2. DIVISION-FREE by design: only multiplies and
// adds, so it needs neither getInvBf16 nor a scalar fp32 divide. (The earlier
// Pade[7/6] form was equally accurate but required a reciprocal; getInvBf16 +
// Newton was the last component never validated on this hardware, and the
// encoder NaN'd with it. Removing the division removes that unknown entirely.)
//
// numpy-validated over [-40, 40] (800k points):
//   max |err| vs true tanh = 6.9e-3 (0.69%)  -- far inside the <4% needed
//   max |out| = 0.996053 < 1  -> gate math bounded, recurrence stable
//   finite everywhere; end-to-end pipeline sim: inflation 1.01x, F1 0.938
static constexpr float kTanhClamp = 3.0f;
static constexpr float kT1 = 9.8983037472e-01f;
static constexpr float kT3 = -2.8898122907e-01f;
static constexpr float kT5 = 7.3146112263e-02f;
static constexpr float kT7 = -1.1429719627e-02f;
static constexpr float kT9 = 9.4607059145e-04f;
static constexpr float kT11 = -3.1460636819e-05f;

// tanh for a single lane, fp32, multiply/add only.
static inline float flair_tanh_f32(float v) {
  // MANDATORY clamp: a polynomial diverges outside its fit interval, so an
  // unclamped evaluation would return |tanh| >> 1 on large gate pre-activations
  // and blow up the recurrence. tanh(3)=0.99505, so clamping costs < 5e-3.
  if (v > kTanhClamp)
    v = kTanhClamp;
  else if (v < -kTanhClamp)
    v = -kTanhClamp;
  const float v2 = v * v;
  float r = kT11;
  r = kT9 + v2 * r;
  r = kT7 + v2 * r;
  r = kT5 + v2 * r;
  r = kT3 + v2 * r;
  r = kT1 + v2 * r;
  return v * r; // |result| <= 0.9961 by construction
}

// sigmoid(x) = 0.5 * (1 + tanh(x/2)). Derived from the tanh above, so the kernel
// contains exactly ONE approximation to validate -- no exp LUT, no raw
// reciprocal. Bounded in (0, 1) by construction.
static inline float flair_sigmoid_f32(float v) {
  return 0.5f * (1.0f + flair_tanh_f32(0.5f * v));
}

// Elementwise GRU gate combine, done entirely in SCALAR fp32:
//   r = sigmoid(gi_r + gh_r);  z = sigmoid(gi_z + gh_z)
//   n = tanh(gi_n + r*gh_n);   h[i] <- (1-z)*n + z*h_old[i]
//
// Deliberately scalar rather than a 16-lane vector loop calling vector
// sigmoid/tanh. That earlier shape needed a store_v -> scalar-loop -> load_v
// round trip per gate (extra stack scratch), and every aie::mul/add/sub on bf16
// vectors returns an aie::accum rather than a vector -- chaining them silently
// yields wrong values in EVERY lane. Going scalar removes that entire failure
// class, drops the scratch arrays, and keeps the polynomial in fp32 (no
// ~0.4%-per-op bf16 rounding inside it). The heavy lifting (matvec_bias) stays
// vectorized, so this costs far less than it looks.
//
// No h_prev COPY is needed: gh was already computed from the FULL old h by the
// caller's matvec_bias, and the combine is elementwise, so h[i]'s old value is
// simply read before it is overwritten on the same iteration.
static inline void gru_gate_combine(const bfloat16 *restrict gi,
                                    const bfloat16 *restrict gh,
                                    bfloat16 *restrict h) {
  constexpr int H = HIDDEN_DIM;
  for (int i = 0; i < H; i++) {
    const float r = flair_sigmoid_f32((float)gi[i] + (float)gh[i]);
    const float z = flair_sigmoid_f32((float)gi[H + i] + (float)gh[H + i]);
    const float n =
        flair_tanh_f32((float)gi[2 * H + i] + r * (float)gh[2 * H + i]);
    const float h_old = (float)h[i]; // read BEFORE the write below
    h[i] = (bfloat16)((1.0f - z) * n + z * h_old);
  }
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
// `h` MUST still be 32-byte (aie::vector_decl_align) aligned and HIDDEN_DIM
// long -- not for the gate combine (which is now scalar), but because
// matvec_bias vector-loads it as `in` (H=64 hits the cols%16==0 path).
// `x_in` is read scalar-only, so it may sit at any offset (e.g. a per-timestep
// slice of a window buffer). `input_dim` = len(x_in) (encoder 45, decoder H).
inline void gru_step(const bfloat16 *restrict x_in, bfloat16 *restrict h,
                     const bfloat16 *restrict w_ih,
                     const bfloat16 *restrict w_hh,
                     const bfloat16 *restrict b_ih,
                     const bfloat16 *restrict b_hh, int input_dim) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  // NOT static. Making these static (to save stack) is WRONG and actively
  // corrupts results: gru_encoder.cc's BATCH loop runs independent windows, so
  // the compiler may software-pipeline/overlap those iterations, and overlapping
  // bodies would clobber shared static scratch. The observed signature was
  // exactly that -- garbage/diverged latents whose failure rate rose
  // monotonically with the slot index within the batch (slot 0: 1/30 ...
  // slot 7: 8/30). Timesteps are serial; batch iterations are NOT.
  //
  // Stack cost is fine: 768B here, LESS than the original kernel's 896B (it also
  // carried h_prev[64], which gru_gate_combine no longer needs).
  alignas(aie::vector_decl_align) bfloat16 gi[H3];
  alignas(aie::vector_decl_align) bfloat16 gh[H3];

  matvec_bias(w_ih, x_in, b_ih, gi, H3, input_dim);
  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses the FULL old h -- must run
                                         // before gru_gate_combine touches h
  gru_gate_combine(gi, gh, h);
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
// `gi` is H3=3*HIDDEN_DIM long and now read scalar-only (the gate combine is
// scalar), so its alignment no longer matters here. `h` MUST still be 32-byte
// aligned and HIDDEN_DIM long -- matvec_bias vector-loads it, same as gru_step.
inline void gru_step_with_gi(const bfloat16 *restrict gi, bfloat16 *restrict h,
                             const bfloat16 *restrict w_hh,
                             const bfloat16 *restrict b_hh) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  // NOT static -- see gru_step: the decoder's batch loop is likewise a set of
  // independent iterations the compiler may overlap, and shared static scratch
  // would be clobbered across them. 384B on the stack is fine.
  alignas(aie::vector_decl_align) bfloat16 gh[H3];

  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses the FULL old h -- must run
                                         // before gru_gate_combine touches h
  gru_gate_combine(gi, gh, h);
}

} // namespace flair

#endif // FLAIR_GRU_COMMON_H
