//===- batch_infer.cpp ------------------------------------*- C++ -*-===//
//
// Generic batched NPU host for a 2-input / 1-output FLAIR kernel (encoder or
// decoder). Loads the xclbin + the shared params buffer ONCE, then streams N
// per-window inputs through the kernel, writing all outputs to one file. This
// amortizes the (expensive) xclbin load across the whole dataset instead of
// paying it per window like the single-shot test.cpp flow.
//
// bf16 buffers are moved as raw uint16 bytes (no host-side interpretation);
// the Python orchestrator (run_dataset_inference.py) does all float math.
//
// Usage:
//   batch_infer.exe <xclbin> <insts> <in1_file> <in2_file> <out_file>
//                   <N> <in1_vol> <in2_vol> <out_vol> [kernel_name]
//     in1_file : N * in1_vol  bf16  (per-window input; e.g. x_window or h0)
//     in2_file : in2_vol      bf16  (shared params, loaded once)
//     out_file : N * out_vol  bf16  (per-window output; e.g. latent or hidden)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using u16 = std::uint16_t; // bf16 storage (raw bits)

static std::vector<u16> load_u16(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("could not open " + path);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<u16> v(n / sizeof(u16));
  if (!f.read(reinterpret_cast<char *>(v.data()), n))
    throw std::runtime_error("could not read " + path);
  return v;
}

int main(int argc, char **argv) {
  if (argc < 10) {
    std::cerr << "usage: batch_infer <xclbin> <insts> <in1> <in2> <out> <N> "
                 "<in1_vol> <in2_vol> <out_vol> [kernel]\n";
    return 1;
  }
  std::string xclbin = argv[1], insts = argv[2];
  std::string in1_file = argv[3], in2_file = argv[4], out_file = argv[5];
  int N = std::stoi(argv[6]);
  int in1_vol = std::stoi(argv[7]);
  int in2_vol = std::stoi(argv[8]);
  int out_vol = std::stoi(argv[9]);
  std::string kernel_name = (argc > 10) ? argv[10] : "MLIR_AIE";

  std::vector<u16> in1 = load_u16(in1_file); // N * in1_vol
  std::vector<u16> in2 = load_u16(in2_file); // in2_vol (shared params)
  if ((int)in1.size() < N * in1_vol || (int)in2.size() < in2_vol) {
    std::cerr << "input file smaller than expected (in1=" << in1.size()
              << " need " << (long)N * in1_vol << ", in2=" << in2.size()
              << " need " << in2_vol << ")\n";
    return 1;
  }
  std::vector<u16> out(static_cast<size_t>(N) * out_vol, 0);

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(insts);

  xrt::device device;
  xrt::kernel kernel;
  test_utils::init_xrt_load_kernel(device, kernel, 0, xclbin, kernel_name);

  // Buffers (group_ids match the 2-in/1-out kernel arg order used by
  // xrt_test_wrapper: 1=instr, 3=in1, 4=in2, 5=out, 6=ctrlpkts, 7=trace).
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, in1_vol * sizeof(u16), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, in2_vol * sizeof(u16), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, out_vol * sizeof(u16), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));
  auto bo_ctrl = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Instr + shared params loaded once.
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  memcpy(bo_in2.map<void *>(), in2.data(), in2_vol * sizeof(u16));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  u16 *map_in1 = bo_in1.map<u16 *>();
  u16 *map_out = bo_out.map<u16 *>();
  const unsigned opcode = 3;

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    memcpy(map_in1, &in1[(size_t)i * in1_vol], in1_vol * sizeof(u16));
    bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out,
                      bo_ctrl, bo_trace);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(&out[(size_t)i * out_vol], map_out, out_vol * sizeof(u16));

    if (((i + 1) % 100) == 0 || i + 1 == N)
      std::cout << "  processed " << (i + 1) << "/" << N << " windows\r"
                << std::flush;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count() / 1000.0;
  std::cout << "\n" << N << " windows in " << ms << " ms ("
            << (ms * 1000.0 / N) << " us/window incl. host overhead)\n";

  std::ofstream fo(out_file, std::ios::binary);
  fo.write(reinterpret_cast<const char *>(out.data()),
           out.size() * sizeof(u16));
  std::cout << "wrote " << out_file << " (" << out.size() << " bf16)\n";
  return 0;
}
