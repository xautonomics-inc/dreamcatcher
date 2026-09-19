#!/bin/bash
# D3 band calibration: banded-vs-banded CROSS-KERNEL drift — ben 2026-09-18.
# The one number toshi needs to size D3's envelope gate tighter than the masked-vs-banded band.
#
#   (1) noah's mainline build, BANDED path (default -fa, exactly as D0 was recorded), --no-repack,
#       compute-mode KLD vs the banded D0 base  -> pure kernel-set divergence on the banded path.
#   (2) optional cross-check: this ik build with -rtr vs its own plain masked logits -> the same
#       measurement on the masked path in the other lineage.
#
# Needs the window (152 GB model). Stops dsv4-flash (:8091, noah's backend) ONLY if it is idle, and
# ALWAYS restores it from the EXIT trap — even if the calling session dies. Oracle dir is never a
# write target: both runs read sha-guarded working copies. Fails closed on every precondition.
set -u
D2=/fast/build/agents/ben/d2-inkling
M=/fast/models/unsloth/Inkling-Small-GGUF/UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00001-of-00005.gguf
W=/fast/build/llama.cpp-inkling/eval-q4km/wikitext-2-raw/wiki.test.raw
NOAH=/fast/build/agents/noah/p1/build/bin/llama-perplexity
MINE=$D2/build-cuda/bin/llama-perplexity
ORIG=/fast/build/agents/noah/p1/d0-oracle-20260915T2024/kld-base-4x2048.bin
WANT=aeab11429a2567db2b36f210754a444ce97f6a9fa851bf0389b236129838b505
BASE=$D2/kld-base-banded.bin                                   # working copy of the banded base
MYMASKED=$D2/d2-parity-20260918T2151/d2-kld-4x2048.bin          # my plain masked logits (parity run)
MYCOPY=$D2/kld-mine-masked-copy.bin
OUT=$D2/d3-calib-$(date -u +%Y%m%dT%H%M); mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""
STOPPED=0
restore() { if [ $STOPPED -eq 1 ]; then systemctl start dsv4-flash.service; for i in $(seq 1 60); do curl -sf -m 4 http://127.0.0.1:8091/health >/dev/null 2>&1 && break; sleep 5; done; echo "$(date -u +%T) RESTORE dsv4-flash: /health $(curl -s -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8091/health)"; fi; }
trap restore EXIT
sha() { sha256sum "$1" | cut -d" " -f1; }
guard() { echo "  guard $1: oracle=$([ "$(sha $ORIG)" = "$WANT" ] && echo OK || echo CHANGED)  copy=$([ "$(sha $BASE)" = "$WANT" ] && echo OK || echo CHANGED)"; }
summ() { grep -aE "Δp|top p|Same top|KLD" "$1" | grep -avE "^llama_|^llm_|^ggml_|^print_info|^load" | tail -16; }
exec > >(tee -a "$OUT/d3-calib.log") 2>&1
echo "=== D3 CALIB $(date -u +%FT%TZ) host=$(hostname) out=$OUT"
# ---- preconditions, BEFORE touching anything ----
for f in "$M" "$W" "$NOAH" "$MINE" "$ORIG" "$MYMASKED"; do [ -e "$f" ] || { echo "PRECONDITION-FAIL: missing $f"; exit 2; }; done
[ "$(sha $ORIG)" = "$WANT" ] || { echo "PRECONDITION-FAIL: oracle base sha mismatch — do not proceed"; exit 2; }
pgrep -x cc1plus >/dev/null && { echo "PRECONDITION-FAIL: compiler running"; exit 2; }
$NOAH --help 2>&1 | grep -q -- "--no-repack" || { echo "PRECONDITION-FAIL: noah build has no --no-repack"; exit 2; }
# In-flight guard via /health. This server's /slots JSON has NO is_processing field, so the old
# check summed a nonexistent key and reported 0 even mid-generation (false zero, found 23:2x).
# /health reports slots_processing directly. Require idle twice, 10 s apart, to shrink the race.
inflight() {
  curl -s -m 5 http://127.0.0.1:8091/health 2>/dev/null | python3 -c 'import sys, json
try:
    print(json.load(sys.stdin).get("slots_processing", "unknown"))
except Exception:
    print("unknown")'
}
p1=$(inflight); sleep 10; p2=$(inflight)
echo "in-flight on :8091 (/health slots_processing): $p1 then $p2"
{ [ "$p1" = "0" ] && [ "$p2" = "0" ]; } || { echo "PRECONDITION-FAIL: :8091 not verified idle twice ($p1/$p2) - nothing stopped"; exit 5; }
# ---- open the window ----
systemctl stop dsv4-flash.service && STOPPED=1 && echo "$(date -u +%T) dsv4-flash stopped (in-flight was 0)"
for i in $(seq 1 18); do avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$avail" -ge 160 ] && break; sleep 5; done
echo "MemAvailable: ${avail} GiB"; [ "$avail" -ge 160 ] || { echo "ABORT: floor not reached"; exit 6; }
cp -p "$ORIG" "$BASE"; cp -p "$MYMASKED" "$MYCOPY"; guard pre
# ---- (1) banded-vs-banded cross-kernel: noah build, banded, no repack, vs banded base ----
echo "=== (1) noah build, BANDED (default -fa), --no-repack, KLD vs banded base  $(date -u +%T)"
$NOAH -m "$M" -ngl 0 -t 20 -f "$W" -c 2048 -b 2048 --chunks 4 --no-repack \
  --kl-divergence --kl-divergence-base "$BASE" > "$OUT/kld-banded-norepack-vs-banded.log" 2>&1
echo "rc=$?"; summ "$OUT/kld-banded-norepack-vs-banded.log"; guard post-1
grep -m1 -aoE "system_info:.*" "$OUT/kld-banded-norepack-vs-banded.log" | grep -oE "(REPACK|LLAMAFILE) = [01]" | tr '\n' ' '; echo
# ---- (2) cross-check: my ik build, masked, -rtr vs my own plain masked logits ----
echo "=== (2) ik build, MASKED, -rtr, KLD vs my plain masked logits  $(date -u +%T)"
$MINE -m "$M" -ngl 0 -t 20 -f "$W" -c 2048 -b 2048 --chunks 4 -fa off -rtr \
  --kl-divergence --kl-divergence-base "$MYCOPY" > "$OUT/kld-rtr-vs-plain-masked.log" 2>&1
echo "rc=$?"; summ "$OUT/kld-rtr-vs-plain-masked.log"; guard post-2
echo "=== D3 CALIB DONE $(date -u +%FT%TZ) out=$OUT"
touch "$OUT/done"
