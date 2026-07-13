//===- gru_decoder.cc -----------------------------------
//
// FLAIR decoder GRU sequence kernel.
//
// Input:
//   h0_vec      : bf16[HIDDEN_DIM]
//   params      : bf16 decoder GRU params packed as:
//                 [w_ih | w_hh | b_ih | b_hh]
// Output:
//   hidden_seq  : bf16[SEQ_LEN * HIDDEN_DIM]
//
// This does only the decoder GRU sequence for now:
//   h = h0_vec
//   for t in 0..SEQ_LEN-1:
//       h = GRUCell(h0_vec, h)
//       hidden_seq[t] = h
//
// Later we add:
//   hidden_seq -> hidden_to_output -> x_hat_num -> MSE
//
//===------------------------------------------------------

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"
#include "gru_common.h"

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

#ifndef SEQ_LEN
#define SEQ_LEN 10
#endif

// Number of windows processed per kernel invocation. params (weights) are
// resident and shared across the whole batch -- only h0_vec/hidden_seq grow
// with BATCH. Defaults to 1 (identical to the original single-window
// behavior) so existing single-window callers are unaffected.
#ifndef BATCH
#define BATCH 1
#endif

extern "C" void gru_decoder_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;

    // Decoder GRU input_dim is HIDDEN_DIM because x_t = h0_vec.
    constexpr int INPUT_DIM = HIDDEN_DIM;

    // Packed params layout, shared across all BATCH windows:
    // [w_ih | w_hh | b_ih | b_hh]
    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *hidden_seq_b = hidden_seq + b * SEQ_LEN * H;

        // Hidden state must be aligned because gru_step vector-loads/stores h.
        alignas(aie::vector_decl_align) bfloat16 h[H];

        // Initial decoder hidden state:
        // h_prev = h0_vec_b
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        // Full decoder GRU sequence.
        for (int t = 0; t < SEQ_LEN; t++) {
            // Decoder input is repeated every timestep:
            // x_t = h0_vec_b
            flair::gru_step(
                h0_vec_b,
                h,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                INPUT_DIM
            );

            // Save h_t into hidden_seq_b[t].
            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = h[i];
            }
        }
    }
}