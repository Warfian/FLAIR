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

  for (int i = 9; i < HIDDEN_DIM; i++)
    out[i] = (bfloat16)0.0f;

  event1();
}

} // extern "C"
