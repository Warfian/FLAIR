//===- gru_cell.cc ----------------------------------------*- C++ -*-===//
//
// FLAIR encoder GRU-cell kernel: one timestep, one AIE core, bf16.
//
// Formula matches PyTorch's nn.GRUCell exactly (validated against a real
// trained checkpoint in npu/verify_gru_cell_math.py -- that script is the
// spec this kernel implements):
//
//   gi = W_ih @ x_in   + b_ih          (3*HIDDEN_DIM,)
//   gh = W_hh @ h_prev + b_hh          (3*HIDDEN_DIM,)
//   r  = sigmoid(gi_r + gh_r)
//   z  = sigmoid(gi_z + gh_z)
//   n  = tanh(gi_n + r * gh_n)         <- gh_n must be gated by r BEFORE
//                                          combining with gi_n; do not sum
//                                          gi_n + gh_n before applying r.
//   h_next = (1 - z) * n + z * h_prev
//
// Gate order along the 3*HIDDEN_DIM axis is [reset, update, new], matching
// PyTorch's weight_ih_l0 / weight_hh_l0 layout, so encoder.gru's state_dict
// tensors can be fed to this kernel with no reordering.
//
// Correctness-first pass: the two matvecs (matvec_bias) are plain scalar
// loops, not yet vectorized -- see aie_kernels/aie2/mv.cc's
// matvec_vectorized for the transposed-layout pattern this should graduate
// to once validated on hardware. The activation/combine stage is
// vectorized in 16-lane bf16 chunks (HIDDEN_DIM=64 -> 4 chunks) using the
// getTanhBf16 LUT primitive from lut_based_ops.h, the same primitive
// aie_kernels/aie2/silu.cc and gelu.cc use; sigmoid is derived from it via
// sigmoid(x) = 0.5*(tanh(x/2)+1), the identical identity silu.cc uses.
//
// NOT YET COMPILED/RUN ON HARDWARE -- this is a first draft. Known risk
// areas to check first: local-array vector-load alignment (gi/gh below
// are declared alignas(aie::vector_decl_align) to address this, but
// unverified), and whether aie::sub is the correct AIE API spelling in
// the installed aie_api version.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These are resolved via -I include dirs passed by the IRON driver
// (gru_cell_encoder.py), NOT via relative paths: the JIT copies this
// source into build/<name>.prj/ before compiling, so a path like
// "../../aie_kernel_utils.h" would not resolve. The driver adds the wheel's
// aie_kernels dir (for aie_kernel_utils.h) and aie_runtime_lib/AIE2 dir
// (for lut_based_ops.h) to the include path, and compiles lut_based_ops.cpp
// into the same TU so getTanhBf16's tanh_lut tables are defined.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

#ifndef INPUT_DIM
#define INPUT_DIM 45
#endif
#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

static_assert(HIDDEN_DIM % 16 == 0,
             "HIDDEN_DIM must be a multiple of 16 (getTanhBf16 operates on "
             "16-lane bf16 vectors)");

namespace {

// sigmoid(x) = 0.5*(tanh(x/2) + 1), via the LUT-based tanh primitive.
// Same identity used by aie_kernels/aie2/silu.cc's sigmoid_approx step.
//
// NOTE: aie::mul / aie::add of two bf16 vectors return an aie::accum, not a
// vector. Each result is assigned to an explicit aie::vector<bfloat16,16>
// (triggering the accum->vector conversion) BEFORE being used as an argument
// to the next aie op -- nesting them (e.g. aie::add(v, aie::mul(...)))
// fails to compile because no overload matches (vector, accum). This
// materialize-every-step idiom mirrors aie_kernels/aie2/silu.cc.
inline aie::vector<bfloat16, 16>
sigmoid16(const aie::vector<bfloat16, 16> &x) {
  aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  aie::vector<bfloat16, 16> half_x = aie::mul(x, half);
  aie::vector<bfloat16, 16> t = getTanhBf16(half_x);
  aie::vector<bfloat16, 16> t_plus_one = aie::add(t, one);
  aie::vector<bfloat16, 16> result = aie::mul(t_plus_one, half);
  return result;
}

// out[row] = sum_i(w[row*cols + i] * in[i]) + bias[row]
// Scalar, float32 accumulation for precision, bf16 storage. Correctness
// first; not vectorized yet.
void matvec_bias(const bfloat16 *restrict w, const bfloat16 *restrict in,
                 const bfloat16 *restrict bias, bfloat16 *restrict out,
                 int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    float acc = (float)bias[row];
    const bfloat16 *restrict w_row = w + row * cols;
    for (int i = 0; i < cols; i++) {
      acc += (float)w_row[i] * (float)in[i];
    }
    out[row] = (bfloat16)acc;
  }
}

} // namespace

// params is a single flat buffer holding all four weight/bias tensors back
// to back, in this order: w_ih (H3*INPUT_DIM), w_hh (H3*HIDDEN_DIM),
// b_ih (H3), b_hh (H3). Bundling into one buffer (rather than four separate
// kernel arguments / ObjectFifos) matches the convention used throughout
// programming_examples/ml (e.g. conv2d.py, scale_shift.py) for constant
// weight tensors, and keeps the IRON driver's rt.sequence() arity small.
void gru_cell_encoder_impl(bfloat16 *restrict x_in, bfloat16 *restrict h_prev,
                           bfloat16 *restrict params,
                           bfloat16 *restrict h_next) {
  event0();

  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;

  bfloat16 *restrict w_ih = params;
  bfloat16 *restrict w_hh = params + H3 * INPUT_DIM;
  bfloat16 *restrict b_ih = w_hh + H3 * H;
  bfloat16 *restrict b_hh = b_ih + H3;

  alignas(aie::vector_decl_align) bfloat16 gi[H3];
  alignas(aie::vector_decl_align) bfloat16 gh[H3];

  matvec_bias(w_ih, x_in, b_ih, gi, H3, INPUT_DIM);
  matvec_bias(w_hh, h_prev, b_hh, gh, H3, H);

  bfloat16 *gi_r = gi;
  bfloat16 *gi_z = gi + H;
  bfloat16 *gi_n = gi + 2 * H;
  bfloat16 *gh_r = gh;
  bfloat16 *gh_z = gh + H;
  bfloat16 *gh_n = gh + 2 * H;

  AIE_LOOP_MIN_ITERATION_COUNT(H / 16)
  for (int i = 0; i < H; i += 16) {
    // See sigmoid16's note: every aie::mul/add/sub result is materialized
    // into an explicit vector before being fed to the next aie op.
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
    // r gates gh_n BEFORE combining with gi_n -- see file header note.
    aie::vector<bfloat16, 16> r_gh_n = aie::mul(r, vgh_n);
    aie::vector<bfloat16, 16> n_pre = aie::add(vgi_n, r_gh_n);
    aie::vector<bfloat16, 16> n = getTanhBf16(n_pre);

    aie::vector<bfloat16, 16> vh_prev = aie::load_v<16>(h_prev + i);
    aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
    aie::vector<bfloat16, 16> one_minus_z = aie::sub(one, z);
    aie::vector<bfloat16, 16> term1 = aie::mul(one_minus_z, n);
    aie::vector<bfloat16, 16> term2 = aie::mul(z, vh_prev);
    aie::vector<bfloat16, 16> h_out = aie::add(term1, term2);

    aie::store_v(h_next + i, h_out);
  }

  event1();
}

extern "C" {

// state buffer = [x_in (INPUT_DIM) | h_prev (HIDDEN_DIM)] concatenated.
// x_in and h_prev are bundled into one input so the compute tile stays
// within its 2-input DMA-channel budget: x_in + h_prev + params as three
// separate input ObjectFifos would need 3 input channels, but each AIE
// core tile has only 2 in / 2 out. state + params = 2 inputs, h_next = 1
// output, which fits.
void gru_cell_encoder_bf16(bfloat16 *state, bfloat16 *params,
                           bfloat16 *h_next) {
  bfloat16 *x_in = state;
  bfloat16 *h_prev = state + INPUT_DIM;
  gru_cell_encoder_impl(x_in, h_prev, params, h_next);
}

} // extern "C"
