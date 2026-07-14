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
// Nonlinearities: Pade[7/6] rational tanh, with sigmoid derived from it.
//
// REPLACES the old exp-LUT path (getExpBf16 + a raw getInvBf16 reciprocal).
// That path was only ~8-9% accurate, and it was the ENTIRE NPU accuracy loss:
// it inflated the normal-p99 detection threshold 6x and cost ~4 F1 points
// (NPU 0.894 vs PyTorch 0.936). Crucially, bf16 rounding itself costs NOTHING
// (1.00x inflation, F1 identical to PyTorch) -- so the nonlinearity was the
// whole error budget. See npu/precision_ablation.py, which reproduces all of
// this without hardware.
//
// Validated in numpy over 2e6 points on [-30, 30]:
//   * max |err| vs true tanh = 6.6e-4   (3.3e-3 after bf16 rounding, vs ~8-9%
//     for the old LUT -- a ~25x improvement, well inside the <4% needed)
//   * den >= 1 ALWAYS  -> the reciprocal is always finite -> CANNOT emit NaN
//   * |tanh| <= 0.999344 < 1 -> gate math stays bounded, GRU stays stable
//   * end-to-end pipeline sim: threshold inflation 6.06x -> 1.01x, F1 -> 0.938
//
// THREE deliberate choices below. Each avoids a failure mode that has ALREADY
// been hit on this kernel -- please do not "simplify" them away:
//
//  1. THE +-4 CLAMP IS MANDATORY. The Pade rational does NOT saturate: its
//     leading terms give num/den -> 0.0357*x as |x| -> inf, so an UNCLAMPED
//     evaluation returns |tanh| > 1 on large gate pre-activations and
//     destabilises the recurrence. tanh(4)=0.99933, so clamping costs < 7e-4.
//
//  2. SCALAR fp32, *not* aie:: vector ops. Per the notes at the top of this
//     file, aie::mul/add/sub on bf16 vectors return an aie::accum, NOT a
//     vector; a chained rational expression like aie::mul(aie::add(a,b),c)
//     silently produces WRONG VALUES IN EVERY LANE -- exactly the "all 64
//     latent dims mismatch" symptom a previous Pade attempt hit. The gate
//     combine (gru_gate_combine, below) is therefore a plain scalar loop: it
//     sidesteps the accum/vector trap entirely, needs no store_v/load_v scratch
//     arrays, and keeps the ~0.4%-per-op bf16 rounding out of the polynomial.
//     matvec_bias stays vectorized, so almost none of the FLOPs are lost.
//
//  3. RECIPROCAL = getInvBf16 SEED + 2 NEWTON STEPS, not a bare '/'. Newton
//     needs only multiply/subtract (certainly available) and squares the
//     relative error each step, so the result is accurate even though the raw
//     getInvBf16 seed is coarse -- and that coarse seed is itself a prime
//     suspect for the old path's error. (Plain `num / den` is equivalent if the
//     toolchain provides scalar fp32 divide; this form simply does not depend
//     on that, and avoids emitting a soft-float call that would cost stack.)
//
// STACK: this costs only a handful of fp32 temporaries -- no scratch arrays at
// all. Combined with moving gi/gh to static (see gru_step), the ~1KB core stack
// has ample headroom, which is what killed the first attempt at this change.
// ---------------------------------------------------------------------------

// Pade[7/6] tanh coefficients, normalised by 135135 so every intermediate stays
// O(1). (The textbook form has coefficients up to 135135 and x^6 terms, which
// would burn bf16 exponent range for no benefit.)
//   tanh(x) ~ x*(1 + a1 x^2 + a2 x^4 + a3 x^6) / (1 + b1 x^2 + b2 x^4 + b3 x^6)
static constexpr float kPadeA1 = 0.1282051282f; // 17325/135135
static constexpr float kPadeA2 = 0.0027972028f; //   378/135135
static constexpr float kPadeA3 = 7.4000740e-6f; //     1/135135
static constexpr float kPadeB1 = 0.4615384615f; // 62370/135135
static constexpr float kPadeB2 = 0.0233100233f; //  3150/135135
static constexpr float kPadeB3 = 2.0720207e-4f; //    28/135135
static constexpr float kPadeClamp = 4.0f;

// 1/d for d >= 1. Coarse seed + 2 Newton-Raphson steps: r <- r*(2 - d*r).
// The relative error squares each step, so even a poor seed converges. Callers
// guarantee d >= 1, so this can never divide by zero and never returns NaN.
static inline float flair_recip(float d) {
  float r = (float)getInvBf16(d);
  r = r * (2.0f - d * r);
  r = r * (2.0f - d * r);
  return r;
}

// Pade tanh for a single lane, evaluated in fp32.
static inline float flair_tanh_f32(float v) {
  if (v > kPadeClamp)
    v = kPadeClamp; // MANDATORY: the Pade does not saturate (see note 1)
  else if (v < -kPadeClamp)
    v = -kPadeClamp;
  const float v2 = v * v;
  const float num = v * (1.0f + v2 * (kPadeA1 + v2 * (kPadeA2 + v2 * kPadeA3)));
  const float den = 1.0f + v2 * (kPadeB1 + v2 * (kPadeB2 + v2 * kPadeB3));
  return num * flair_recip(den); // den >= 1 -> always finite, |result| < 1
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

  // STATIC (L1/BSS), NOT the stack. These are 768B and the AIE core stack is
  // only ~1KB: 768B of gi/gh plus the scalar gate loop's fp32 temporaries
  // overflow it, which corrupts memory and yields garbage/NaN latents. That is
  // a CORRUPTION NaN, not an arithmetic one -- the tell is that it moves with
  // code layout (see ACCURACY_HANDOFF.md Sec 6, and scalar_probe.cc, which
  // isolated exactly this).
  //
  // Safe to share across calls: timesteps are strictly serial (step t+1 needs h
  // from step t) and batch items run one after another, so two gru_step bodies
  // can never overlap. matvec_bias rewrites all H3 rows of gi/gh every call.
  alignas(aie::vector_decl_align) static bfloat16 gi[H3];
  alignas(aie::vector_decl_align) static bfloat16 gh[H3];

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

  // STATIC (L1/BSS), not the stack -- same core-stack reasoning as gru_step
  // above (384B here). This is this function's own static, distinct from
  // gru_step's.
  alignas(aie::vector_decl_align) static bfloat16 gh[H3];

  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses the FULL old h -- must run
                                         // before gru_gate_combine touches h
  gru_gate_combine(gi, gh, h);
}

} // namespace flair

#endif // FLAIR_GRU_COMMON_H
