//===- scalar_probe.cc ------------------------------------*- C++ -*-===//
//
// Diagnostic probe: isolate whether complex SCALAR fp32 arithmetic works on
// this AIE core. Runs a ladder of increasingly complex scalar fp32 ops on a
// fixed input (x = 1.0) and writes each intermediate to the output. Whichever
// output first reads NaN pinpoints the failing operation.
//
// Reuses the encoder harness buffer signature (x_window, params, out) so no
// new host/Makefile plumbing is needed; x_window/params are ignored. Read the
// per-index "got" values from the verbose test output (they all "mismatch" the
// encoder golden, so all print).
//
// Expected (finite) results if scalar fp32 works:
//   out[0]=1        out[1]=1        out[2]=2
//   out[3]~135168   out[4]~152576   out[5]~6.56e-6   out[6]~6.56e-6
//   out[7]~0.7616 (tanh 1)          out[8]~0.7311 (sigmoid 1)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h> // getInvBf16
#include <stdint.h>

using namespace aie;

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif
#ifndef INPUT_DIM
#define INPUT_DIM 45
#endif

// scalar fp32 reciprocal: getInvBf16 seed + 2 Newton steps (mul/sub only).
static inline float recip(float d) {
  float r = (float)getInvBf16(d);
  r = r * (2.0f - d * r);
  r = r * (2.0f - d * r);
  return r;
}

// scalar fp32 Pade[7/6] tanh.
static inline float tanh_approx(float x) {
  if (x > 4.0f)
    x = 4.0f;
  else if (x < -4.0f)
    x = -4.0f;
  float x2 = x * x;
  float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
  return num * recip(den);
}

static inline float sigmoid_approx(float x) {
  return 0.5f * (1.0f + tanh_approx(0.5f * x));
}

extern "C" {

void scalar_probe_bf16(bfloat16 *x_window, bfloat16 *params, bfloat16 *out) {
  event0();
  (void)params;

  // RUNTIME input (from the DMA'd buffer) so the compiler cannot constant-fold
  // the ladder -- this exercises the actual runtime scalar-fp32 path the
  // encoder uses. x_window[0] is real data; read out[0] to see its value and
  // verify out[7]==tanh(out[0]), out[8]==sigmoid(out[0]). Finite outputs =>
  // runtime scalar fp32 works; NaN => it's broken (the encoder's real cause).
  float x = (float)x_window[0];
  float x2 = x * x;
  float d = 135135.0f + x2 * 62370.0f; // runtime, ~197505 for x=1

  out[0] = (bfloat16)x;                           // runtime sanity
  out[1] = (bfloat16)(x * x);                     // runtime scalar mul
  out[2] = (bfloat16)(x + x);                     // runtime scalar add
  out[3] = (bfloat16)(135135.0f * x);             // runtime large value
  out[4] = (bfloat16)(135135.0f + x2 * 17325.0f); // runtime large-const arith
  out[5] = (bfloat16)((float)getInvBf16(d));      // runtime getInvBf16(large)
  out[6] = (bfloat16)recip(d);                    // runtime recip + Newton
  out[7] = (bfloat16)tanh_approx(x);              // runtime full Pade tanh
  out[8] = (bfloat16)sigmoid_approx(x);           // runtime full sigmoid
  out[9] = (bfloat16)0.0f;                         // (gap marker)

  // --- Integration test: matvec output feeding a scalar gate ---
  // Layout of params (real encoder weights): w_ih | w_hh | b_ih | b_hh.
  // Compute gi[0] = w_ih[0,:] . x_window[0:INPUT_DIM] + b_ih[0] (the first
  // matvec output row), then feed it through the scalar sigmoid. This is
  // exactly what gru_step does; if out[10..12] are finite, matvec->gate
  // integration works and the encoder NaN is in the recurrence/loop.
  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;
  const bfloat16 *w_ih = params;
  const bfloat16 *b_ih = params + H3 * INPUT_DIM + H3 * H;

  float gi0 = (float)b_ih[0];
  for (int i = 0; i < INPUT_DIM; i++)
    gi0 += (float)w_ih[i] * (float)x_window[i]; // matvec row 0

  out[10] = (bfloat16)gi0;                  // matvec output (runtime)
  out[11] = (bfloat16)tanh_approx(gi0);     // Pade tanh on matvec output
  out[12] = (bfloat16)sigmoid_approx(gi0);  // sigmoid on matvec output

  // --- Full single gru_step test (reproduces the encoder's per-step stack:
  // gi[192] + gh[192] arrays + the 64-unit scalar Pade gate loop) ---
  const bfloat16 *w_hh = params + H3 * INPUT_DIM;
  const bfloat16 *b_hh = b_ih + H3;

  // static -> L1 data (BSS), NOT the ~1 KB stack. The 768 B of gi/gh on the
  // stack + the scalar loop temporaries overflow it (the NaN moves with code
  // layout -> memory corruption, not a data-dependent NaN).
  static bfloat16 gi[H3];
  static bfloat16 gh[H3];
  bfloat16 hloc[H];
  for (int i = 0; i < H; i++)
    hloc[i] = (bfloat16)0.0f; // h = 0

  for (int row = 0; row < H3; row++) {
    float acc = (float)b_ih[row];
    for (int i = 0; i < INPUT_DIM; i++)
      acc += (float)w_ih[row * INPUT_DIM + i] * (float)x_window[i];
    gi[row] = (bfloat16)acc;
  }
  for (int row = 0; row < H3; row++) {
    float acc = (float)b_hh[row];
    for (int i = 0; i < H; i++)
      acc += (float)w_hh[row * H + i] * (float)hloc[i];
    gh[row] = (bfloat16)acc;
  }
  for (int i = 0; i < H; i++) {
    float r = sigmoid_approx((float)gi[i] + (float)gh[i]);
    float z = sigmoid_approx((float)gi[H + i] + (float)gh[H + i]);
    float n = tanh_approx((float)gi[2 * H + i] + r * (float)gh[2 * H + i]);
    float h_old = (float)hloc[i];
    hloc[i] = (bfloat16)((1.0f - z) * n + z * h_old);
  }

  // Output the first 8 hidden values of the full step to out[13..20].
  for (int i = 0; i < 8; i++)
    out[13 + i] = hloc[i];

  // --- Unit-6 breakdown (the NaN unit): dump every intermediate so we see
  // exactly which gate input / nonlinearity produces the NaN. ---
  {
    int u = 6;
    float pre_r = (float)gi[u] + (float)gh[u];
    float r6 = sigmoid_approx(pre_r);
    float pre_z = (float)gi[H + u] + (float)gh[H + u];
    float z6 = sigmoid_approx(pre_z);
    float n_pre = (float)gi[2 * H + u] + r6 * (float)gh[2 * H + u];
    float n6 = tanh_approx(n_pre);
    out[21] = (bfloat16)pre_r;  // reset-gate input
    out[22] = (bfloat16)r6;     // sigmoid(pre_r)
    out[23] = (bfloat16)pre_z;  // update-gate input
    out[24] = (bfloat16)z6;     // sigmoid(pre_z)
    out[25] = (bfloat16)n_pre;  // candidate input
    out[26] = (bfloat16)n6;     // tanh(n_pre)
    out[27] = (bfloat16)((float)gi[2 * H + u]); // raw gi_n[6]
    out[28] = (bfloat16)((float)gh[2 * H + u]); // raw gh_n[6]
  }

  for (int i = 29; i < HIDDEN_DIM; i++)
    out[i] = (bfloat16)0.0f;

  event1();
}

} // extern "C"
