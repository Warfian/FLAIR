//===- gru_encoder.cc -------------------------------------*- C++ -*-===//
//
// FLAIR encoder: a full T-timestep GRU encode in ONE kernel invocation, one
// AIE core, bf16. Weights are loaded once (resident in L1) and the hidden
// state is carried on-core across all T timesteps -- this is the key change
// vs invoking the single-cell kernel T times (which re-DMA'd the 42 KB
// weights and paid host/launch overhead every timestep).
//
//   h = 0
//   for t in 0..T-1:  h = GRUCell(x_window[t], h)
//   latent = h            (last hidden state)
//
// Reuses the hardware-validated gru_step from gru_common.h.
//
// Buffers (2 inputs + 1 output, within the core's 2-in/2-out DMA budget):
//   x_window : (T * INPUT_DIM) bf16   per-timestep inputs, concatenated
//   params   : encoder GRU weights [w_ih | w_hh | b_ih | b_hh], resident
//   latent   : (HIDDEN_DIM) bf16      output (last hidden state)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

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
#ifndef SEQ_LEN
#define SEQ_LEN 10
#endif

#include "gru_common.h"

void gru_encoder_impl(bfloat16 *restrict x_window, bfloat16 *restrict params,
                      bfloat16 *restrict latent) {
  event0();

  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;
  constexpr int T = SEQ_LEN;

  // params layout: w_ih (H3*INPUT_DIM) | w_hh (H3*H) | b_ih (H3) | b_hh (H3)
  const bfloat16 *restrict w_ih = params;
  const bfloat16 *restrict w_hh = params + H3 * INPUT_DIM;
  const bfloat16 *restrict b_ih = w_hh + H3 * H;
  const bfloat16 *restrict b_hh = b_ih + H3;

  // Resident hidden state, initialized to zero, carried across timesteps.
  alignas(aie::vector_decl_align) bfloat16 h[H];
  for (int i = 0; i < H; i++)
    h[i] = (bfloat16)0.0f;

  for (int t = 0; t < T; t++) {
    // x_window[t] is a length-INPUT_DIM slice at element offset t*INPUT_DIM.
    // gru_step reads x_in scalar-only, so this unaligned offset is fine.
    const bfloat16 *restrict x_t = x_window + t * INPUT_DIM;
    flair::gru_step(x_t, h, w_ih, w_hh, b_ih, b_hh, INPUT_DIM);
  }

  for (int i = 0; i < H; i++)
    latent[i] = h[i];

  event1();
}

extern "C" {

void gru_encoder_bf16(bfloat16 *x_window, bfloat16 *params, bfloat16 *latent) {
  gru_encoder_impl(x_window, params, latent);
}

} // extern "C"
