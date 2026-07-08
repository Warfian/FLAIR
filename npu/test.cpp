//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the FLAIR GRU-cell NPU kernel. Runs on native Windows
// (where XRT sees the NPU); built via CMake/Visual Studio and invoked from
// the WSL Makefile through powershell.exe.
//
// Loads three binary files produced by gen_test_data.py (in the run dir):
//   gru_state.bin   : bf16 [x_in | h_prev | pad]   (STATE_LEN elems)
//   gru_params.bin  : bf16 [w_ih | w_hh | b_ih | b_hh] (N_PARAMS elems)
//   gru_golden.bin  : float32 reference h_next       (HIDDEN_DIM elems)
// feeds state + params to the NPU, reads back h_next, and compares against
// the golden with a tolerance (bf16 rounding + LUT tanh/sigmoid make exact
// equality impossible).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

using bf16 = test_utils::bfloat16_t;

// Absolute tolerance for the bf16 + LUT-approx NPU output vs the float golden.
#ifndef VERIFY_ATOL
#define VERIFY_ATOL 0.05f
#endif

//*****************************************************************************
// File-loaded reference data (populated in main() before setup_and_run_aie).
//*****************************************************************************

static std::vector<bf16> g_state;   // STATE_LEN bf16
static std::vector<bf16> g_params;  // N_PARAMS bf16
static std::vector<float> g_golden; // HIDDEN_DIM float32

template <typename T>
static std::vector<T> load_bin(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("could not open " + path);
  std::streamsize n_bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<T> out(n_bytes / sizeof(T));
  if (!f.read(reinterpret_cast<char *>(out.data()), n_bytes))
    throw std::runtime_error("could not read " + path);
  return out;
}

//*****************************************************************************
// Buffer init / verify hooks for the 2-input / 1-output wrapper.
//*****************************************************************************

void initialize_bufIn_state(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = (i < (int)g_state.size()) ? g_state[i] : bf16(0.0f);
}

void initialize_bufIn_params(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = (i < (int)g_params.size()) ? g_params[i] : bf16(0.0f);
}

void initialize_bufOut(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = bf16(0.0f);
}

// NOTE: the wrapper passes IN1_VOLUME (the state length) as `SIZE`, not the
// output length -- so we loop over HIDDEN_DIM (the real h_next length) here.
// out[i] is a raw bf16 (uint16 bits on MSVC); convert with bfloat16_to_float,
// NOT static_cast (which would give the integer bit value).
int verify_gru(bf16 *state, bf16 *params, bf16 *out, int /*SIZE*/,
               int verbosity) {
  (void)state;
  (void)params;
  int errors = 0;
  float max_abs = 0.0f;
  for (int i = 0; i < HIDDEN_DIM; i++) {
    float got = test_utils::bfloat16_to_float(out[i]);
    float ref = g_golden[i];
    float d = std::abs(got - ref);
    if (std::isfinite(got) && d > max_abs)
      max_abs = d;
    if (!std::isfinite(got) || d > VERIFY_ATOL) {
      errors++;
      if (verbosity >= 1)
        std::cout << "  mismatch[" << i << "]: got " << got << " (0x"
                  << std::hex << static_cast<unsigned>(out[i]) << std::dec
                  << ") ref " << ref << " (|d|=" << d << ")\n";
    }
  }
  std::cout << "h_next max abs error vs golden (finite): " << max_abs
            << "  (atol=" << VERIFY_ATOL << ", " << errors << "/" << HIDDEN_DIM
            << " over tol)\n";
  return errors;
}

//*****************************************************************************

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(bf16); // state
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(bf16); // params
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(bf16); // h_next

  g_state = load_bin<bf16>("gru_state.bin");
  g_params = load_bin<bf16>("gru_params.bin");
  g_golden = load_bin<float>("gru_golden.bin");

  if ((int)g_state.size() < IN1_VOLUME || (int)g_params.size() < IN2_VOLUME ||
      (int)g_golden.size() < HIDDEN_DIM) {
    std::cerr << "input file(s) smaller than expected; regenerate with "
                 "gen_test_data.py\n";
    return 1;
  }

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<bf16, bf16, bf16, initialize_bufIn_state,
                              initialize_bufIn_params, initialize_bufOut,
                              verify_gru>(IN1_VOLUME, IN2_VOLUME, OUT_VOLUME,
                                          myargs, false);
  return res;
}
