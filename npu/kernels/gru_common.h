//===- gru_common.h ---------------------------------------*- C++ -*-===//
//
// Shared AIE kernel building blocks for the FLAIR GRU designs (encoder,
// decoder, full forward). Scalar, correctness-first (vectorizing the matvec
// is a later phase).
//
// Nonlinearities use a rational Pade approximation of tanh in fp32 (arithmetic
// only, ~6e-4 accurate and unbiased -- numpy-validated to hold the 10-step
// encode to <0.005 vs the float golden). The exp-LUT path (getExpBf16) was
// only ~12.8% accurate and its bias compounded through the recurrence.
// sigmoid(x) = 0.5*(1 + tanh(x/2)).
//
// AIE hardware constraints that shaped this file:
//  * The core stack is tiny (~1 KB). The gate pre-activations gi/gh are 192
//    bf16 each (768 B together); on the stack, they + the scalar gate loop's
//    temporaries overflow it and corrupt memory (manifests as a NaN whose
//    index MOVES with code layout). They are declared `static` so they live
//    in L1 data (BSS), NOT the stack. Safe because they are pure scratch,
//    fully written before read on every gru_step call.
//  * The core has no fp32 divide (`a / b` yields NaN); reciprocal uses
//    getInvBf16 (a proven bit-manipulation reciprocal from lut_based_ops.h)
//    refined by Newton steps.
//
// Include requirement: the including .cc must pull in aie_api/aie.hpp (bfloat16
// type) and lut_based_ops.h (getInvBf16) before this header, and the driver
// must compile lut_based_ops.cpp into the TU (for m_inv_lut).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef FLAIR_GRU_COMMON_H
#define FLAIR_GRU_COMMON_H

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

namespace flair {

// 1/d for d > 0 without the '/' operator (the AIE core has no fp32 divide).
// getInvBf16 (pointer-cast bit-manipulation reciprocal, proven to run on this
// core) as the seed, refined by two Newton steps (r <- r*(2 - d*r)).
inline float recip(float d) {
  float r = (float)getInvBf16(d);
  r = r * (2.0f - d * r);
  r = r * (2.0f - d * r);
  return r;
}

// Rational Pade[7/6] approximation of tanh, fp32. ~6e-4 for |x| <= 4; clamp
// beyond (tanh saturates: tanh(4) ~ 0.9993). Arithmetic + recip only.
inline float tanh_approx(float x) {
  if (x > 4.0f)
    x = 4.0f;
  else if (x < -4.0f)
    x = -4.0f;
  float x2 = x * x;
  float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
  return num * recip(den);
}

inline float sigmoid_approx(float x) {
  return 0.5f * (1.0f + tanh_approx(0.5f * x));
}

// out[row] = sum_i(w[row*cols + i] * in[i]) + bias[row]. fp32 accumulation,
// bf16 output. Scalar. `bias` may be nullptr.
inline void matvec_bias(const bfloat16 *restrict w, const bfloat16 *restrict in,
                        const bfloat16 *restrict bias, bfloat16 *restrict out,
                        int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    float acc = bias ? (float)bias[row] : 0.0f;
    const bfloat16 *restrict w_row = w + row * cols;
    for (int i = 0; i < cols; i++)
      acc += (float)w_row[i] * (float)in[i];
    out[row] = (bfloat16)acc;
  }
}

// One GRU timestep, updating the bf16 hidden state h in place (PyTorch
// nn.GRUCell math):
//   gi = w_ih @ x_in + b_ih ; gh = w_hh @ h + b_hh    (bf16 pre-activations)
//   r = sigmoid(gi_r+gh_r) ; z = sigmoid(gi_z+gh_z)
//   n = tanh(gi_n + r*gh_n) ; h <- (1-z)*n + z*h       (fp32 gates + combine)
// Gate order [reset, update, new] (PyTorch weight_ih/hh layout).
// `input_dim` is the length of x_in (encoder: 45, decoder: HIDDEN_DIM).
inline void gru_step(const bfloat16 *restrict x_in, bfloat16 *restrict h,
                     const bfloat16 *restrict w_ih,
                     const bfloat16 *restrict w_hh,
                     const bfloat16 *restrict b_ih,
                     const bfloat16 *restrict b_hh, int input_dim) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  // static -> L1 data (BSS), off the ~1 KB core stack (see file header).
  // Pure scratch: fully written by the matvecs before the gate loop reads it,
  // so sharing one instance across gru_step calls is safe.
  static bfloat16 gi[H3];
  static bfloat16 gh[H3];

  matvec_bias(w_ih, x_in, b_ih, gi, H3, input_dim);
  matvec_bias(w_hh, h, b_hh, gh, H3, H); // uses old h

  for (int i = 0; i < H; i++) {
    float r = sigmoid_approx((float)gi[i] + (float)gh[i]);
    float z = sigmoid_approx((float)gi[H + i] + (float)gh[H + i]);
    float n = tanh_approx((float)gi[2 * H + i] + r * (float)gh[2 * H + i]);
    float h_old = (float)h[i];
    h[i] = (bfloat16)((1.0f - z) * n + z * h_old);
  }
}

} // namespace flair

#endif // FLAIR_GRU_COMMON_H
