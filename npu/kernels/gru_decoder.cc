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

extern "C" void gru_decoder_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;

    // Decoder GRU input_dim is HIDDEN_DIM because x_t = h0_vec.
    constexpr int INPUT_DIM = HIDDEN_DIM;

    // Packed params layout:
    // [w_ih | w_hh | b_ih | b_hh]
    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    // Hidden state must be aligned because gru_step vector-loads/stores h.
    alignas(aie::vector_decl_align) bfloat16 h[H];

    // Initial decoder hidden state:
    // h_prev = h0_vec
    for (int i = 0; i < H; i++) {
        h[i] = h0_vec[i];
    }

    // Full decoder GRU sequence.
    for (int t = 0; t < SEQ_LEN; t++) {
        // Decoder input is repeated every timestep:
        // x_t = h0_vec
        flair::gru_step(
            h0_vec,
            h,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            INPUT_DIM
        );

        // Save h_t into hidden_seq[t].
        for (int i = 0; i < H; i++) {
            hidden_seq[t * H + i] = h[i];
        }
    }
}