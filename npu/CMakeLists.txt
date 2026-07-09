# CMake host-build config for the FLAIR GRU-cell NPU test harness.
# Modeled on mlir-aie's programming_examples/basic/vector_scalar_mul/CMakeLists.txt.
#
# Parameters (passed via -D by the Makefile or on the cmake command line):
#   -DXRT_INC_DIR : XRT headers (src/runtime_src/core/include in the XRT repo)
#   -DXRT_LIB_DIR : dir containing xrt_coreutil.lib
#   -DTARGET_NAME : output executable base name
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Dims from the trained checkpoint (INPUT_DIM=45, HIDDEN_DIM=64, SEQ_LEN=10).
# Encoder buffers: in1 = x_window (SEQ_LEN*INPUT_DIM), in2 = params
# (encoder GRU = 3*HIDDEN_DIM*(INPUT_DIM+HIDDEN_DIM) + 2*3*HIDDEN_DIM = 21312),
# out = latent (HIDDEN_DIM).
set(INPUT_DIM 45 CACHE STRING "GRU input dim")
set(HIDDEN_DIM 64 CACHE STRING "GRU hidden dim")
set(SEQ_LEN 10 CACHE STRING "encoder window length")
set(N_PARAMS 21312 CACHE STRING "total packed encoder params")

# Buffer sizes in bytes (bf16 = 2 bytes/elem).
math(EXPR IN1_SIZE "${SEQ_LEN} * ${INPUT_DIM} * 2")
math(EXPR IN2_SIZE "${N_PARAMS} * 2")
math(EXPR OUT_SIZE "${HIDDEN_DIM} * 2")

# This design lives at programming_examples/FLAIR_Dan/npu -- two levels under
# programming_examples/, same depth as the basic/ examples, so the shared
# cmake helpers are two levels up.
include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
    INPUT_DIM=${INPUT_DIM}
    HIDDEN_DIM=${HIDDEN_DIM}
    SEQ_LEN=${SEQ_LEN}
    N_PARAMS=${N_PARAMS}
    IN1_SIZE=${IN1_SIZE}
    IN2_SIZE=${IN2_SIZE}
    OUT_SIZE=${OUT_SIZE}
)

target_include_directories(${currentTarget} PUBLIC ${XRT_INC_DIR})
target_link_directories(${currentTarget} PUBLIC ${XRT_LIB_DIR})
target_link_libraries(${currentTarget} PUBLIC xrt_coreutil)
target_link_test_utils(${currentTarget})
