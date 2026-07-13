cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(INPUT_DIM 45 CACHE STRING "GRU input dim")
set(HIDDEN_DIM 64 CACHE STRING "GRU hidden dim")
set(SEQ_LEN 10 CACHE STRING "sequence length")
set(N_PARAMS 21312 CACHE STRING "packed params")
set(NPU_MODE encoder CACHE STRING "encoder or decoder")

if("${NPU_MODE}" STREQUAL "encoder")
    set(HOST_SRC test.cpp)
    math(EXPR IN1_SIZE "${SEQ_LEN} * ${INPUT_DIM} * 2")
    math(EXPR IN2_SIZE "${N_PARAMS} * 2")
    math(EXPR OUT_SIZE "${HIDDEN_DIM} * 2")
elseif("${NPU_MODE}" STREQUAL "decoder")
    set(HOST_SRC test_decoder.cpp)
    math(EXPR IN1_SIZE "${HIDDEN_DIM} * 2")
    math(EXPR IN2_SIZE "${N_PARAMS} * 2")
    math(EXPR OUT_SIZE "${SEQ_LEN} * ${HIDDEN_DIM} * 2")
elseif("${NPU_MODE}" STREQUAL "batch")
    # Generic batched host; buffer sizes come from argv, not compile-time.
    set(HOST_SRC batch_infer.cpp)
    set(IN1_SIZE 2)
    set(IN2_SIZE 2)
    set(OUT_SIZE 2)
else()
    message(FATAL_ERROR "NPU_MODE must be encoder, decoder, or batch")
endif()

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} ${HOST_SRC})

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
