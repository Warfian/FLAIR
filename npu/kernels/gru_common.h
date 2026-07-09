//===- gru_common.h ---------------------------------------*- C++ -*-===//
//
// Shared AIE kernel building blocks for the FLAIR GRU designs (encoder,
// decoder, full forward). Scalar, fp32-accurate, correctness-first
// (vectorization of the matvec is a later phase).
//
// Nonlinearities use a rational Pade approximation of tanh in fp32 (arithmetic
// only -- no LUT). The earlier exp-LUT path (getExpBf16) is only ~12.8%
// accurate and systematically biased, which compounded through the recurrence
// (encoder latent diverged up to ~0.35 over 10 steps). The Pade tanh is
// ~6e-4 accurate and unbiased; validated in numpy to hold the 10-step encode
// to <0.005 vs the float golden. sigmoid(x) = 0.5*(1 + tanh(x/2)).
//
// The gates + combine run in fp32; the hidden state is stored bf16 between
// timesteps (numpy-confirmed sufficient once the gates are accurate). The
// matvecs accumulate in fp32 and emit fp32 gate pre-activations.
//
// Include requirement: the including .cc must pull in aie_api/aie.hpp (for the
// bfloat16 type) before this header. No lut_based_ops.h dependency.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef FLAIR_GRU_COMMON_H
#define FLAIR_GRU_COMMON_H

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

namespace flair {

// Rational Pade[7/6] approximation of tanh, fp32. Accurate to ~6e-4 for
// |x| <= 4; clamp beyond (tanh saturates: tanh(4) ~ 0.9993). Arithmetic only.
inline float tanh_approx(float x) {
  if (x > 4.0f)
    x = 4.0f;
  else if (x < -4.0f)
    x = -4.0f;
  float x2 = x * x;
  float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
  return num / den;
}

inline float sigmoid_approx(float x) {
  return 0.5f * (1.0f + tanh_approx(0.5f * x));
}

// out[row] = sum_i(w[row*cols + i] * in[i]) + bias[row], fp32 accumulation,
// fp32 output. Scalar (not yet vectorized). `bias` may be nullptr.
inline void matvec_bias_f32out(const bfloat16 *restrict w,
                               const bfloat16 *restrict in,
                               const bfloat16 *restrict bias,
                               float *restrict out, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    float acc = bias ? (float)bias[row] : 0.0f;
    const bfloat16 *restrict w_row = w + row * cols;
    for (int i = 0; i < cols; i++)
      acc += (float)w_row[i] * (float)in[i];
    out[row] = acc;
  }
}

// One GRU timestep, updating the bf16 hidden state h in place (PyTorch
// nn.GRUCell math):
//   gi = w_ih @ x_in + b_ih ; gh = w_hh @ h + b_hh   (fp32)
//   r = sigmoid(gi_r+gh_r) ; z = sigmoid(gi_z+gh_z)
//   n = tanh(gi_n + r*gh_n) ; h <- (1-z)*n + z*h      (fp32 gates + combine)
// Gate order [reset, update, new] (PyTorch weight_ih/hh layout).
// `input_dim` is the length of x_in (encoder: 45, decoder: HIDDEN_DIM).
// No alignment requirement on h (all scalar access).
inline void gru_step(const bfloat16 *restrict x_in, bfloat16 *restrict h,
                     const bfloat16 *restrict w_ih,
                     const bfloat16 *restrict w_hh,
                     const bfloat16 *restrict b_ih,
                     const bfloat16 *restrict b_hh, int input_dim) {
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  float gi[H3];
  float gh[H3];
  matvec_bias_f32out(w_ih, x_in, b_ih, gi, H3, input_dim);
  matvec_bias_f32out(w_hh, h, b_hh, gh, H3, H); // uses old h

  for (int i = 0; i < H; i++) {
    float r = sigmoid_approx(gi[i] + gh[i]);
    float z = sigmoid_approx(gi[H + i] + gh[H + i]);
    float n = tanh_approx(gi[2 * H + i] + r * gh[2 * H + i]);
    float h_old = (float)h[i];
    h[i] = (bfloat16)((1.0f - z) * n + z * h_old);
  }
}

} // namespace flair

#endif // FLAIR_GRU_COMMON_H
