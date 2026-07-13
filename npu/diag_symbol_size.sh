#!/usr/bin/env bash
# diag_symbol_size.sh -- DIAGNOSTIC ONLY
#
# Quick check: does __attribute__((noinline)) on gru_step_with_gi shrink
# gru_decoder_bf16 back down toward gru_encoder_impl's size (1152B), and
# does gru_step_with_gi's own reported size grow back toward gru_step's
# (6336B, since it's no longer able to fold into the caller)?
#
# Usage (from npu/): bash diag_symbol_size.sh
set -uo pipefail

rm -rf build/decoder.prj
python3 gru_decoder.py --dev npu --hidden-dim 64 --seq-len 10 --batch 6 \
  --xclbin-path build/decoder.xclbin --insts-path build/decoder_insts.bin

echo "================================================================"
echo "gru_decoder.prj symbol sizes (nm --print-size --size-sort)"
echo "================================================================"
nm --print-size --size-sort -C build/decoder.prj/gru_decoder_bf16.o 2>/dev/null | tail -10
