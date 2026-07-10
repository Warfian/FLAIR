#include "xrt_test_wrapper.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using bf16 = test_utils::bfloat16_t;

#ifndef VERIFY_ATOL
#define VERIFY_ATOL 0.10f
#endif

static std::vector<bf16> g_h0;
static std::vector<bf16> g_params;
static std::vector<float> g_golden;

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

void initialize_bufIn_h0(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = (i < static_cast<int>(g_h0.size())) ? g_h0[i] : bf16(0.0f);
}

void initialize_bufIn_params(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = (i < static_cast<int>(g_params.size())) ? g_params[i] : bf16(0.0f);
}

void initialize_bufOut(bf16 *buf, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    buf[i] = bf16(0.0f);
}

int verify_decoder(bf16 *h0, bf16 *params, bf16 *out, int /*SIZE*/,
                   int verbosity) {
  (void)h0;
  (void)params;

  int errors = 0;
  float max_abs = 0.0f;
  double sum_abs = 0.0;

  constexpr int TOTAL = SEQ_LEN * HIDDEN_DIM;

  for (int i = 0; i < TOTAL; i++) {
    int t = i / HIDDEN_DIM;
    int h = i % HIDDEN_DIM;

    float got = test_utils::bfloat16_to_float(out[i]);
    float ref = g_golden[i];
    float d = std::abs(got - ref);

    if (std::isfinite(got) && d > max_abs)
      max_abs = d;

    if (std::isfinite(got))
      sum_abs += d;

    if (!std::isfinite(got) || d > VERIFY_ATOL) {
      errors++;
      if (verbosity >= 1 && errors <= 20) {
        std::cout << "  mismatch[t=" << t << ", h=" << h << "]: got " << got
                  << " ref " << ref << " |d|=" << d << "\n";
      }
    }
  }

  float mean_abs = static_cast<float>(sum_abs / TOTAL);

  std::cout << "decoder hidden_seq max abs error vs golden: " << max_abs
            << "  mean abs error: " << mean_abs << "  atol=" << VERIFY_ATOL
            << "  errors=" << errors << "/" << TOTAL << "\n";

  if (errors == 0)
    std::cout << "PASS: decoder hidden_seq within tolerance.\n";
  else
    std::cout << "FAIL: decoder hidden_seq has mismatches.\n";

  return errors;
}

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(bf16);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(bf16);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(bf16);

  g_h0 = load_bin<bf16>("decoder_h0.bin");
  g_params = load_bin<bf16>("decoder_gru_params.bin");
  g_golden = load_bin<float>("decoder_hidden_golden.bin");

  if ((int)g_h0.size() < IN1_VOLUME || (int)g_params.size() < IN2_VOLUME ||
      (int)g_golden.size() < OUT_VOLUME) {
    std::cerr << "decoder input file(s) smaller than expected; regenerate with "
                 "gen_decoder_data.py\n";
    return 1;
  }

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<bf16, bf16, bf16, initialize_bufIn_h0,
                              initialize_bufIn_params, initialize_bufOut,
                              verify_decoder>(IN1_VOLUME, IN2_VOLUME,
                                              OUT_VOLUME, myargs, false);
  return res;
}


