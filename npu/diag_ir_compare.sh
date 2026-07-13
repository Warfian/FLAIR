#!/usr/bin/env bash
# diag_ir_compare.sh -- DIAGNOSTIC ONLY
#
# Compares the compiled LLVM IR (.opt.ll, post-optimization -- closest to
# what actually gets scheduled onto the AIE core) across the encoder and the
# decoder diagnostic builds, to find why the SAME sigmoid/tanh gate-combine
# loop (identical source in gru_common.h) costs ~85% of the decoder's
# compute time but is cheap in the encoder.
#
# Run from npu/ after diag_decoder_timing.py has built all 4 decoder
# variants (so their .prj dirs exist) and gru_encoder.py has built the
# encoder at some point (build/gru.prj should already exist from earlier
# runs -- rebuild it first if not: python3 gru_encoder.py --dev npu
# --input-dim 48 --hidden-dim 64 --seq-len 10 --batch 6 --xclbin-path
# build/gru.xclbin --insts-path build/insts.bin).
#
# Usage: bash diag_ir_compare.sh
set -uo pipefail

PROJECTS=(
  "encoder:build/gru.prj"
  "decoder_unfused:build/decoder.prj"
  "decoder_matvec_only:build/decoder_matvec_only.prj"
  "decoder_noop:build/decoder_noop.prj"
)

echo "================================================================"
echo "1. File presence + size"
echo "================================================================"
for entry in "${PROJECTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry#*:}"
  ll="$dir/main_core_0_2.opt.ll"
  if [ -f "$ll" ]; then
    lines=$(wc -l < "$ll")
    bytes=$(wc -c < "$ll")
    echo "  $name: $ll  ($lines lines, $bytes bytes)"
  else
    echo "  $name: $ll  NOT FOUND (build this variant first)"
  fi
done

echo
echo "================================================================"
echo "2. Symbol/keyword frequency (case-insensitive) per file"
echo "   Looking for: sigmoid, tanh, inv, exp, getinv, getexp, call, br "
echo "   (br = branch instructions; more 'br label %X' with backward"
echo "   targets suggests a real loop; a flat count with no backward"
echo "   branches suggests full unrolling)"
echo "================================================================"
for entry in "${PROJECTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry#*:}"
  ll="$dir/main_core_0_2.opt.ll"
  [ -f "$ll" ] || continue
  echo "--- $name ---"
  for kw in sigmoid tanh inv exp getinv getexp; do
    n=$(grep -ic "$kw" "$ll" 2>/dev/null || echo 0)
    echo "    $kw: $n"
  done
  ncalls=$(grep -c '^\s*call ' "$ll" 2>/dev/null || echo 0)
  nbr=$(grep -c '\bbr \b' "$ll" 2>/dev/null || echo 0)
  ndefs=$(grep -c '^define' "$ll" 2>/dev/null || echo 0)
  echo "    total 'call' instrs: $ncalls"
  echo "    total 'br' instrs:   $nbr"
  echo "    function definitions ('define'): $ndefs"
  echo
done

echo "================================================================"
echo "3. Function definitions list (names only) per file"
echo "   (shows whether sigmoid16/tanh16/getInvBf16 appear as SEPARATE"
echo "   functions -- not inlined -- or don't appear at all -- inlined"
echo "   away already at this stage)"
echo "================================================================"
for entry in "${PROJECTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry#*:}"
  ll="$dir/main_core_0_2.opt.ll"
  [ -f "$ll" ] || continue
  echo "--- $name ---"
  grep -oE '^define[^{]*@[A-Za-z0-9_.]+' "$ll" | sed -E 's/^define[^@]*@/  /' | head -30
  echo
done

echo "================================================================"
echo "4. Instruction-type histogram (top 15) per file"
echo "   (opcode counts -- compares WHAT KIND of work dominates)"
echo "================================================================"
for entry in "${PROJECTS[@]}"; do
  name="${entry%%:*}"
  dir="${entry#*:}"
  ll="$dir/main_core_0_2.opt.ll"
  [ -f "$ll" ] || continue
  echo "--- $name ---"
  # crude opcode extraction: first token after '= ' or leading keyword
  grep -oE '^\s*(%[A-Za-z0-9_.]+\s*=\s*)?[a-z]+' "$ll" \
    | sed -E 's/^\s*(%[A-Za-z0-9_.]+\s*=\s*)?//' \
    | sort | uniq -c | sort -rn | head -15 \
    | awk '{printf "    %6d  %s\n", $1, $2}'
  echo
done

echo "================================================================"
echo "Done. Paste sections 1-4 back for comparison."
echo "================================================================"
