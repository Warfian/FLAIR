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

// INPUT_DIM is the PADDED input length (real 45 features + 3 zero pad = 48),
// a multiple of 16 so the w_ih matvec vectorizes. The padded weights/inputs
// are zero, so the result is unchanged. The driver passes -DINPUT_DIM=48.
#ifndef INPUT_DIM
#define INPUT_DIM 48
#endif
#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 10
#endif
// Number of windows processed per kernel invocation. params (weights) are
// resident and shared across the whole batch -- only x_window/latent grow
// with BATCH. Defaults to 1 (identical to the original single-window
// behavior) so existing single-window callers are unaffected.
#ifndef BATCH
#define BATCH 1
#endif

#include "gru_common.h"

void gru_encoder_impl(bfloat16 *restrict x_window, bfloat16 *restrict params,
                      bfloat16 *restrict latent) {
  event0();

  constexpr int H = HIDDEN_DIM;
  constexpr int H3 = 3 * H;
  constexpr int T = SEQ_LEN;

  // params layout: w_ih (H3*INPUT_DIM) | w_hh (H3*H) | b_ih (H3) | b_hh (H3)
  // Shared across all BATCH windows -- loaded once, read BATCH times.
  const bfloat16 *restrict w_ih = params;
  const bfloat16 *restrict w_hh = params + H3 * INPUT_DIM;
  const bfloat16 *restrict b_ih = w_hh + H3 * H;
  const bfloat16 *restrict b_hh = b_ih + H3;

  for (int b = 0; b < BATCH; b++) {
    bfloat16 *restrict x_window_b = x_window + b * T * INPUT_DIM;
    bfloat16 *restrict latent_b = latent + b * H;

    // Resident hidden state, initialized to zero, carried across timesteps.
    alignas(aie::vector_decl_align) bfloat16 h[H];
    for (int i = 0; i < H; i++)
      h[i] = (bfloat16)0.0f;

    // Per-timestep gi (the encoder's x_t CHANGES every step, so gi is NOT
    // invariant -- can't hoist like the decoder). Computed here in the caller,
    // then fed to gru_step_with_gi. This is IDENTICAL math to the old
    // gru_step(x_t, ...) call (gi = w_ih @ x_t + b_ih, then gh + gate), but
    // routes the gate loop through gru_step_with_gi -- the leaner function the
    // decoder uses, which measures far faster on hardware than the monolithic
    // gru_step for the same work (encoder was 417us/window of pure compute,
    // ~165us above a component model; diag_encoder_timing localized it to the
    // gru_step path, not DMA). Peak stack frame is caller gi(384) +
    // gru_step_with_gi gh(384) + h(128) = 896B, same as the old gru_step path.
    alignas(aie::vector_decl_align) bfloat16 gi[H3];
    for (int t = 0; t < T; t++) {
      // x_window_b[t] is a length-INPUT_DIM slice at element offset
      // t*INPUT_DIM. matvec_bias reads x_in scalar/vector; the offset t*48 is
      // 32-byte aligned (48 = 3*16), so the vectorized path is valid.
      const bfloat16 *restrict x_t = x_window_b + t * INPUT_DIM;
      flair::matvec_bias(w_ih, x_t, b_ih, gi, H3, INPUT_DIM);
      flair::gru_step_with_gi(gi, h, w_hh, b_hh);
    }

    for (int i = 0; i < H; i++)
      latent_b[i] = h[i];
  }

  event1();
}

// DIAGNOSTIC ONLY -- does virtually no compute. Same (x_window, params,
// latent) signature and buffer sizes as gru_encoder_bf16, and the SAME
// ObjectFifo/DMA wiring (so x_window + params still get DMA'd in at full
// size), but calls NO gru_step. Purpose: localize the encoder's ~145us
// unexplained per-window overhead. If this shows ~the same low floor as the
// decoder-noop (~32us/window), the overhead is in the gru_step/compute path
// (a codegen problem). If it stays high, the overhead is dispatch/DMA of the
// encoder's large x_window input (15x the decoder's input per dispatch).
void gru_encoder_noop_impl(bfloat16 *restrict x_window, bfloat16 *restrict params,
                           bfloat16 *restrict latent) {
  event0();

  constexpr int H = HIDDEN_DIM;
  constexpr int T = SEQ_LEN;

  // Touch params[0] so the params DMA/read isn't optimized away entirely.
  bfloat16 touch = params[0];

  for (int b = 0; b < BATCH; b++) {
    bfloat16 *restrict x_window_b = x_window + b * T * INPUT_DIM;
    bfloat16 *restrict latent_b = latent + b * H;

    // Write latent from x_window (reads the input buffer so its DMA/load
    // isn't dead, but does NO matvec/gate compute). +touch-touch keeps params
    // live without changing the value.
    for (int i = 0; i < H; i++)
      latent_b[i] = x_window_b[i] + touch - touch;
  }

  event1();
}

extern "C" {

void gru_encoder_bf16(bfloat16 *x_window, bfloat16 *params, bfloat16 *latent) {
  gru_encoder_impl(x_window, params, latent);
}

void gru_encoder_noop_bf16(bfloat16 *x_window, bfloat16 *params,
                           bfloat16 *latent) {
  gru_encoder_noop_impl(x_window, params, latent);
}

} // extern "C"
