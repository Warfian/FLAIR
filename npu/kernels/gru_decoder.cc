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

        // Decoder input x_t = h0_vec_b is IDENTICAL on every timestep (no
        // autoregressive/categorical feedback), so gi = w_ih @ h0_vec_b +
        // b_ih is invariant too -- compute it ONCE instead of every
        // timestep (gru_step would otherwise redo this same matvec 10x).
        alignas(aie::vector_decl_align) bfloat16 gi[H3];
        flair::matvec_bias(w_ih, h0_vec_b, b_ih, gi, H3, INPUT_DIM);

        // Full decoder GRU sequence.
        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step_with_gi(gi, h, w_hh, b_hh);

            // Save h_t into hidden_seq_b[t].
            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = h[i];
            }
        }
    }
}

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 21
#endif

// Fused variant: computes the final reconstruction (hidden_to_output) ON
// the core instead of returning the raw hidden_seq, so the output buffer
// is BATCH*SEQ_LEN*OUTPUT_DIM (420B/window at OUTPUT_DIM=21) instead of
// BATCH*SEQ_LEN*HIDDEN_DIM (1280B/window) -- a 3x smaller per-window output,
// freeing L1 budget for a larger BATCH. Separate entry point from
// gru_decoder_bf16 above so the single-window live-demo/verify flow
// (test_decoder.cpp, gen_decoder_data.py, compare_anomaly_score.py) is
// completely unaffected.
extern "C" void gru_decoder_fused_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *recon
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;
    constexpr int INPUT_DIM = HIDDEN_DIM;
    constexpr int OUT = OUTPUT_DIM;

    // Packed params layout, shared across all BATCH windows:
    // [w_ih | w_hh | b_ih | b_hh | w_out | b_out]
    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;
    bfloat16 *w_out = b_hh + H3;        // (OUT, H)
    bfloat16 *b_out = w_out + OUT * H;  // (OUT,)

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *recon_b = recon + b * SEQ_LEN * OUT;

        alignas(aie::vector_decl_align) bfloat16 h[H];
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step(
                h0_vec_b,
                h,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                INPUT_DIM
            );

            // hidden_to_output, fused: recon_b[t] = w_out @ h + b_out
            flair::matvec_bias(w_out, h, b_out, recon_b + t * OUT, OUT, H);
        }
    }
}

// DIAGNOSTIC ONLY -- not part of the scoring pipeline. Runs the exact same
// full GRU sequence as gru_decoder_bf16, but writes ONLY the final hidden
// state (BATCH*HIDDEN_DIM output, like the encoder's latent) instead of the
// whole hidden_seq (BATCH*SEQ_LEN*HIDDEN_DIM). Purpose: isolate whether the
// decoder's large fixed per-dispatch cost (~5.5x the encoder's) comes from
// the per-timestep output writes / larger output DMA, or from the gru_step
// compute itself. Identical compute to gru_decoder_bf16; only the output
// footprint differs (batch*64 vs batch*640).
extern "C" void gru_decoder_final_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *final_h
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;
    constexpr int INPUT_DIM = HIDDEN_DIM;

    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *final_h_b = final_h + b * H;

        alignas(aie::vector_decl_align) bfloat16 h[H];
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        // Same gi-hoisting as gru_decoder_bf16 -- see that function's comment.
        alignas(aie::vector_decl_align) bfloat16 gi[H3];
        flair::matvec_bias(w_ih, h0_vec_b, b_ih, gi, H3, INPUT_DIM);

        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step_with_gi(gi, h, w_hh, b_hh);
        }

        // Write ONLY the final hidden state (batch*H total).
        for (int i = 0; i < H; i++) {
            final_h_b[i] = h[i];
        }
    }
}

// DIAGNOSTIC ONLY -- does virtually no compute. Same (h0, params,
// hidden_seq) argument signature and buffer sizes as gru_decoder_bf16, and
// the SAME ObjectFifo/acquire-release wiring (so params still gets DMA'd in,
// same ack/release pattern), but calls NO gru_step at all. Purpose: if this
// STILL shows the ~3300us/dispatch floor, the cost is not about on-core
// compute at all -- it's structural to how THIS xclbin gets dispatched
// (tile placement, buffer/DMA setup at compile time, etc.), independent of
// what code runs on the core. Not used for scoring.
extern "C" void gru_decoder_noop_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *hidden_seq_b = hidden_seq + b * SEQ_LEN * H;

        // Touch params (one element) so it isn't compiled away entirely,
        // without doing any real matvec/gate compute.
        bfloat16 touch = params[0];

        for (int i = 0; i < H; i++) {
            hidden_seq_b[i] = h0_vec_b[i] + touch - touch; // == h0_vec_b[i]
        }
        for (int t = 1; t < SEQ_LEN; t++) {
            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = (bfloat16)0.0f;
            }
        }
    }
}