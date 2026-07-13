#!/usr/bin/env bash
# diag_ir_compare2.sh -- DIAGNOSTIC ONLY
#
# Round 1 (diag_ir_compare.sh) compared main_core_0_2.opt.ll and found it's
# just the ObjectFifo/DMA wrapper (identical across all 4 variants, no
# sigmoid/tanh/matvec code in it -- that's compiled as a SEPARATE
# ExternalFunction compilation unit, e.g. gru_decoder_bf16.cc/.o, linked in
# separately). This script finds and inspects THAT artifact instead.
#
# Usage: bash diag_ir_compare2.sh
set -uo pipefail

PROJECTS=(
  "encoder:build/gru.prj"
  "decoder_unfused:build/decoder.prj"
  "decoder_matvec_only:build/decoder_matvec_only.prj"
  "decoder_noop:build/decoder_noop.prj"
)

echo "================================================================"
echo "1. Full directory listing per project (looking for the kernel's"
echo "   own compiled artifacts, e.g. gru_decoder_bf16.cc/.o/.ll)"
echo "================================================================"
for entry in "${PROJECTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry#*:}"
  echo "--- $name ($dir) ---"
  if [ -d "$dir" ]; then
    ls -la "$dir" 2>/dev/null
  else
    echo "  (directory not found)"
  fi
  echo
done

echo "================================================================"
echo "2. Any .ll files anywhere under build/ matching kernel names"
echo "================================================================"
find build -iname "*bf16*.ll" -o -iname "*bf16*.o" 2>/dev/null | sort

echo
echo "================================================================"
echo "3. Tool availability for disassembly fallback"
echo "================================================================"
for tool in llvm-objdump objdump llvm-dis; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "  $tool: $(command -v "$tool")"
  else
    echo "  $tool: not found"
  fi
done

echo
echo "================================================================"
echo "Paste sections 1-3 back. If a *_bf16.o (kernel object, no .ll"
echo "sibling) shows up, next step is disassembling it with whichever"
echo "objdump tool is available."
echo "================================================================"
