#!/usr/bin/env bash
# diag_disasm_compare.sh -- DIAGNOSTIC ONLY
#
# Round 2 (diag_ir_compare2.sh) found the actual kernel body compiles to a
# separate object per ExternalFunction (e.g. gru_decoder_bf16.o), with no
# intermediate LLVM IR preserved -- only main_core_0_2.* (the ObjectFifo/DMA
# wrapper, identical across variants, irrelevant here). This script:
#   1. Force-rebuilds EVERY project from scratch (rm -rf first) -- staleness
#      already caused one false negative today (see memory), don't risk it
#      again, especially since the encoder has never been freshly rebuilt
#      during this whole investigation.
#   2. Per-symbol size comparison (nm) -- which function, in which object,
#      is actually big, independent of total file size (each .o may contain
#      code for functions beyond just its "target" one).
#   3. Disassembly (objdump -d, plain objdump -- llvm-objdump isn't
#      installed) of just the target function in each .o, with instruction
#      counts and a search for calls into getInvBf16/getExpBf16 (inlined vs
#      real call instructions).
#
# Usage (from npu/, WSL IRON env sourced): bash diag_disasm_compare.sh
set -uo pipefail

echo "================================================================"
echo "0. Force clean rebuild of everything (encoder + all decoder variants)"
echo "================================================================"
rm -rf build/gru.prj build/decoder.prj build/decoder_final.prj \
       build/decoder_noop.prj build/decoder_matvec_only.prj

python3 gru_encoder.py --dev npu --input-dim 48 --hidden-dim 64 --seq-len 10 \
  --batch 6 --xclbin-path build/gru.xclbin --insts-path build/insts.bin
python3 gru_decoder.py --dev npu --hidden-dim 64 --seq-len 10 --batch 6 \
  --xclbin-path build/decoder.xclbin --insts-path build/decoder_insts.bin
python3 gru_decoder_noop.py --dev npu --hidden-dim 64 --seq-len 10 --batch 6 \
  --xclbin-path build/decoder_noop.xclbin --insts-path build/decoder_noop_insts.bin
python3 gru_decoder_matvec_only.py --dev npu --hidden-dim 64 --seq-len 10 \
  --batch 6 --xclbin-path build/decoder_matvec_only.xclbin \
  --insts-path build/decoder_matvec_only_insts.bin

TARGETS=(
  "encoder:build/gru.prj/gru_encoder_bf16.o:gru_encoder_bf16"
  "decoder_unfused:build/decoder.prj/gru_decoder_bf16.o:gru_decoder_bf16"
  "decoder_noop:build/decoder_noop.prj/gru_decoder_noop_bf16.o:gru_decoder_noop_bf16"
  "decoder_matvec_only:build/decoder_matvec_only.prj/gru_decoder_matvec_only_bf16.o:gru_decoder_matvec_only_bf16"
)

echo
echo "================================================================"
echo "1. Confirm fresh mtimes (all should be from THIS run, seconds apart)"
echo "================================================================"
for entry in "${TARGETS[@]}"; do
  IFS=: read -r name obj sym <<< "$entry"
  if [ -f "$obj" ]; then
    ls -la --time-style=full-iso "$obj"
  else
    echo "  $name: $obj NOT FOUND"
  fi
done

echo
echo "================================================================"
echo "2. Per-symbol sizes (nm --print-size --size-sort), top 10 per file"
echo "================================================================"
for entry in "${TARGETS[@]}"; do
  IFS=: read -r name obj sym <<< "$entry"
  [ -f "$obj" ] || continue
  echo "--- $name ($obj) ---"
  nm --print-size --size-sort -C "$obj" 2>/dev/null | tail -10
  echo
done

echo "================================================================"
echo "3. Target function size specifically"
echo "================================================================"
for entry in "${TARGETS[@]}"; do
  IFS=: read -r name obj sym <<< "$entry"
  [ -f "$obj" ] || continue
  line=$(nm --print-size -C "$obj" 2>/dev/null | grep " $sym$")
  echo "  $name ($sym): ${line:-not found in symbol table}"
done

echo
echo "================================================================"
echo "4. Disassembly of the target function (objdump -d), instruction"
echo "   count + first/last 20 lines for a structural look"
echo "================================================================"
for entry in "${TARGETS[@]}"; do
  IFS=: read -r name obj sym <<< "$entry"
  [ -f "$obj" ] || continue
  echo "--- $name: disassemble $sym ---"
  out=$(objdump -d --disassemble="$sym" -C "$obj" 2>&1)
  ninsn=$(echo "$out" | grep -cE '^\s+[0-9a-f]+:' || echo 0)
  echo "  instruction lines: $ninsn"
  echo "$out" | grep -E '^\s+[0-9a-f]+:' | head -15
  echo "  ..."
  echo "$out" | grep -E '^\s+[0-9a-f]+:' | tail -10
  echo
done

echo "================================================================"
echo "5. Calls to getInvBf16/getExpBf16 (inlined away vs real call)"
echo "   per target function's disassembly"
echo "================================================================"
for entry in "${TARGETS[@]}"; do
  IFS=: read -r name obj sym <<< "$entry"
  [ -f "$obj" ] || continue
  echo "--- $name ---"
  objdump -d --disassemble="$sym" -C "$obj" 2>&1 \
    | grep -iE 'call|inv|exp' | head -10
  echo
done

echo "================================================================"
echo "Done. Paste sections 1-5. Section 1's mtimes are the freshness"
echo "check; if any look old, something is still stale."
echo "================================================================"
