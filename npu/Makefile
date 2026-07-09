##===- Makefile — FLAIR fused GRU encoder NPU design ---------------------===##
#
# Two-step compile-in-WSL / run-on-Windows flow (buildHostWin.md):
#   (i)  build the AIE design in WSL  -> build/gru.xclbin + build/insts.bin
#   (ii) build test.cpp on Windows via cmake/powershell, then run on the NPU
#
# Active design: gru_encoder (full SEQ_LEN-timestep encode in one invocation).
# The single-cell files (gru_cell.cc / gru_cell_encoder.py / gen_test_data.py)
# remain in the tree as the validated proof-of-concept but are not the active
# target here.
#
# Modeled on programming_examples/basic/vector_scalar_mul/Makefile.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = gru_encoder
aie_py_src = gru_encoder.py

# This is a Ryzen 7940HS (Phoenix / XDNA1 / AIE2) -> device "npu".
devicename = npu

INPUT_DIM ?= 45
HIDDEN_DIM ?= 64
SEQ_LEN ?= 10

# Optional XRT paths for the Windows host build. On native Windows, set these
# to your XRT clone (per buildHostWin.md), e.g.:
#   make run XRT_INC_DIR=/mnt/c/Technical/XRT/src/runtime_src/core/include \
#            XRT_LIB_DIR=/mnt/c/Technical/xrtNPUfromDLL
# If unset, common.cmake tries to auto-detect XRT via find_package(XRT).
XRT_FLAGS =
ifdef XRT_INC_DIR
XRT_FLAGS += -DXRT_INC_DIR=`${getwslpath} ${XRT_INC_DIR}`
endif
ifdef XRT_LIB_DIR
XRT_FLAGS += -DXRT_LIB_DIR=`${getwslpath} ${XRT_LIB_DIR}`
endif

xclbin_target = build/gru.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target} data

# (i) Compile the @iron.jit design to xclbin + insts (WSL, no NPU needed).
$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},--dev ${devicename} --input-dim ${INPUT_DIM} --hidden-dim ${HIDDEN_DIM} --seq-len ${SEQ_LEN}))

# jit_xclbin only lists the .py driver as a prerequisite, so edits to the
# kernel sources would NOT trigger an xclbin rebuild. Add them explicitly.
${xclbin_target} ${insts_target}: ${srcdir}/kernels/gru_encoder.cc ${srcdir}/kernels/gru_common.h

# (ii) Build the host test.exe (Windows, via cmake/powershell).
$(eval $(call build_host_exe,${targetname},_build,test.cpp,${XRT_FLAGS} -DINPUT_DIM=${INPUT_DIM} -DHIDDEN_DIM=${HIDDEN_DIM} -DSEQ_LEN=${SEQ_LEN}))

# Golden + input binaries for the host harness (needs torch; run in WSL).
.PHONY: data
data: gru_golden.bin
gru_state.bin gru_params.bin gru_golden.bin &: gen_encoder_data.py
	python3 ${srcdir}/gen_encoder_data.py

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target} data
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -v 1

.PHONY: clean
clean:
	rm -rf build _build ${targetname}*.exe gru_state.bin gru_params.bin gru_golden.bin
