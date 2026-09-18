#!/bin/bash
# D2 Inkling parity runner — ben 2026-09-17. Counterpart to run-d0-oracle.ben.sh.
#
# Modes:
#   precheck  Settle the banded-FA question CHEAPLY on the 77 GB UD-IQ2_M quant using the
#             INTERNAL build (which has the banded op): one ppl pass with default -fa, one with
#             -fa 0, then diff the four chunk values. Differ => the recorded D0 oracle used
#             banded FA and is the wrong comparand for D2 (which has no banded op); a non-FA
#             oracle must be re-recorded. Identical => the recorded oracle stands.
#   parity    Run THIS tree's D2 build on UD-Q4_K_M and compare to the D0 oracle:
#             64-token greedy token-for-token, and the four per-chunk perplexity values.
#
# PRECONDITION (booked window): resident services quieted by claude-ops; MemAvailable >= 160 GiB.
# Never runs alongside a build. Stops its own processes on exit. Fails closed.
set -uo pipefail

MODE="${1:-parity}"

D2=/fast/build/agents/ben/d2-inkling
B=$D2/build-cuda/bin                      # CUDA-BUILT, run CPU-only: the oracle was built the
                                          # same way (its server.log opens with
                                          # "ggml_cuda_init: failed ... no CUDA-capable device").
NOAH_B=/fast/build/agents/noah/p1/build/bin
MDIR=/fast/models/unsloth/Inkling-Small-GGUF
M_Q4=$MDIR/UD-Q4_K_M/Inkling-Small-UD-Q4_K_M-00001-of-00005.gguf
M_IQ2=$MDIR/UD-IQ2_M/Inkling-Small-UD-IQ2_M-00001-of-00003.gguf
WIKI=/fast/build/llama.cpp-inkling/eval-q4km/wikitext-2-raw/wiki.test.raw
ORACLE=/fast/build/agents/noah/p1/d0-oracle-20260915T2024
OUT=$D2/d2-$MODE-$(date -u +%Y%m%dT%H%M)
THREADS=20
PORT=18098                                # NOT 18097: never collide with a live oracle server
MEM_FLOOR_GIB=160
MEM_ABORT_KB=$((12*1024*1024))

# Inkling has no banded FA in this tree, so build_inkling asserts unless FA is off. The bias is
# per-head and ggml_flash_attn_ext never materialises kq, so there is nothing to add it to.
FA_OFF="-fa 0"

export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"
LOG=$OUT/d2.log
exec > >(tee -a "$LOG") 2>&1

SP=""; WD=""
cleanup() {
  [ -n "$WD" ] && kill "$WD" 2>/dev/null
  if [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null; then kill -TERM "$SP"; wait "$SP" 2>/dev/null; echo "[cleanup] server stopped"; fi
  pkill -f "[l]lama-perplexity -m $MDIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
memlog() { awk -v t="$(date -u +%T)" -v l="$1" '/^MemAvailable:/{printf "%s mem %s: %.1f GiB\n", t, l, $2/1048576}' /proc/meminfo | tee -a "$OUT/memavailable.log"; }
watchdog() {
  ( while :; do
      kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      echo "$(date -u +%T) wd $((kb/1048576)) GiB" >> "$OUT/memavailable.log"
      if [ "$kb" -lt "$MEM_ABORT_KB" ]; then
        echo "$(date -u +%T) WATCHDOG ABORT: MemAvailable $((kb/1048576)) GiB < 12 GiB" | tee -a "$LOG"
        pkill -x llama-server; pkill -x llama-perplexity; exit 1
      fi
      sleep 30
    done ) &
  WD=$!
}

precond() {
  local avail; avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  memlog pre
  [ "$avail" -ge "$MEM_FLOOR_GIB" ] || { echo "PRECONDITION-FAIL: MemAvailable ${avail} GiB < ${MEM_FLOOR_GIB} GiB (services not quieted?)"; exit 2; }
  pgrep -x cc1plus >/dev/null && { echo "PRECONDITION-FAIL: a compiler is running on the host"; exit 2; }
  for f in "$@"; do [ -e "$f" ] || { echo "PRECONDITION-FAIL: missing $f"; exit 2; }; done
}

ppl_run() {  # ppl_run <binary> <model> <extra-args> <logfile>
  "$1" -m "$2" -ngl 0 -t $THREADS -f "$WIKI" -c 2048 -b 2048 --chunks 4 $3 > "$4" 2>&1
  echo "  ppl rc=$? -> $4"
  grep -oE '^\[[0-9]+\][0-9.]+' "$4" | tr -d '[]' | sed 's/^[0-9]*//' | tr '\n' ' '; echo
}

case "$MODE" in
# ---------------------------------------------------------------- precheck
precheck)
  echo "=== D2 PRECHECK (banded-FA question) $(date -u +%FT%TZ) host=$(hostname)"
  echo "    internal build on IQ2_M: default -fa vs -fa 0. Same quant both runs; only the"
  echo "    COMPARISON matters, absolute values are meaningless across quants."
  precond "$M_IQ2" "$WIKI" "$NOAH_B/llama-perplexity"
  watchdog
  echo "--- run A: default -fa (AUTO => flash_attn true => banded path if predicate holds)"
  A=$(ppl_run "$NOAH_B/llama-perplexity" "$M_IQ2" "" "$OUT/ppl-fa-auto.log" | tail -1)
  memlog after-A
  echo "--- run B: -fa 0 (masked path, what D2 does)"
  B2=$(ppl_run "$NOAH_B/llama-perplexity" "$M_IQ2" "-fa 0" "$OUT/ppl-fa-off.log" | tail -1)
  memlog after-B
  echo
  echo "chunks with -fa auto : $A"
  echo "chunks with -fa 0    : $B2"
  if [ "$A" = "$B2" ]; then
    echo "VERDICT: IDENTICAL -> flash-attn path does not change the numbers; the recorded D0"
    echo "         oracle stands as D2's comparand. Proceed to: $0 parity"
  else
    echo "VERDICT: DIFFER -> the recorded oracle is NOT a valid comparand for D2's masked path."
    echo "         Re-record a non-FA oracle for D2; keep the FA capture as D3's gate."
  fi
  ;;
# ---------------------------------------------------------------- parity
parity)
  # Provenance: the staged tree has no .git (it is rsynced without it), so fall back to
  # hashing the binaries actually being run. A parity number without provenance is not evidence.
  TREE=$(git -C $D2 rev-parse --short HEAD 2>/dev/null)
  [ -n "$TREE" ] || TREE="nogit:server=$(sha256sum $B/llama-server | cut -c1-12),ppl=$(sha256sum $B/llama-perplexity | cut -c1-12)"
  echo "=== D2 PARITY $(date -u +%FT%TZ) host=$(hostname) tree=$TREE"
  echo "    oracle=$ORACLE  model=$M_Q4  fa=OFF(masked)  threads=$THREADS"
  precond "$M_Q4" "$WIKI" "$B/llama-server" "$B/llama-perplexity" "$ORACLE/greedy-64x8.json" "$ORACLE/ppl.log"
  watchdog

  # ---- (1) greedy: the oracle drove this through llama-server, NOT llama-cli ----
  $B/llama-server -m "$M_Q4" -ngl 0 -t $THREADS -c 4096 -np 1 --jinja $FA_OFF \
      --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 360); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
    kill -0 $SP 2>/dev/null || { echo "SERVER-DIED"; tail -30 "$OUT/server.log"; exit 3; }
    sleep 5
  done
  curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER-NOT-HEALTHY"; exit 3; }
  memlog server-healthy

  python3 - "$PORT" "$OUT" "$ORACLE" <<'PY'
import json,sys,urllib.request
port,out,oracle=sys.argv[1],sys.argv[2],sys.argv[3]
ref=json.load(open(f"{oracle}/greedy-64x8.json"))
res=[];fails=0
for rec in ref:
    p=rec["prompt"]
    body={"prompt":p,"n_predict":64,"temperature":0,"top_k":1,"seed":1,
          "cache_prompt":False,"return_tokens":True,"n_probs":10}
    r=json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}),timeout=1800))
    got,want=r.get("tokens"),rec.get("tokens")
    ok = got==want
    if not ok:
        fails+=1
        n=min(len(got or []),len(want or []))
        first=next((i for i in range(n) if got[i]!=want[i]), n)
        print(f"prompt {rec['i']}: MISMATCH at token {first} (got {got[first:first+4] if got else None} want {want[first:first+4]})",flush=True)
    else:
        print(f"prompt {rec['i']}: MATCH ({len(got)} tokens)",flush=True)
    res.append({"i":rec["i"],"match":ok,"tokens":got,
                "completion_probabilities":r.get("completion_probabilities")})
json.dump(res,open(f"{out}/d2-greedy-64x8.json","w"),indent=1)
print(f"GREEDY: {len(ref)-fails}/{len(ref)} prompts token-for-token identical")
sys.exit(1 if fails else 0)
PY
  grc=$?; echo "greedy compare rc=$grc"
  kill -TERM $SP; wait $SP 2>/dev/null; SP=""; echo "$(date -u +%T) server stopped"
  memlog after-greedy

  # ---- (2) perplexity vs the oracle's four chunk values ----
  ppl_run "$B/llama-perplexity" "$M_Q4" "$FA_OFF --kl-divergence-base $OUT/d2-kld-4x2048.bin" "$OUT/ppl.log"
  memlog after-ppl
  got=$(grep -oE '^\[[0-9]+\][0-9.]+' "$OUT/ppl.log" | sed 's/^\[[0-9]*\]//' | tr '\n' ' ')
  want=$(grep -oE '^\[[0-9]+\][0-9.]+' "$ORACLE/ppl.log" | sed 's/^\[[0-9]*\]//' | tr '\n' ' ')
  echo "ppl oracle : $want"
  echo "ppl d2     : $got"
  prc=0; [ "$got" = "$want" ] || prc=1

  ( cd "$OUT" && sha256sum d2-greedy-64x8.json ppl.log > SHA256SUMS ) 2>/dev/null
  echo
  if [ $grc -eq 0 ] && [ $prc -eq 0 ]; then
    echo "=== D2 PARITY PASS (greedy token-for-token AND 4/4 ppl chunks) out=$OUT"
  else
    echo "=== D2 PARITY FAIL (greedy rc=$grc, ppl rc=$prc) out=$OUT"
    echo "    Before blaming the port: confirm the oracle's flash-attn path matches this run's."
    echo "    D2 runs masked (-fa 0); if the oracle ran banded FA the comparand is wrong."
    echo "    Settle with: $0 precheck"
  fi
  [ $grc -eq 0 ] && [ $prc -eq 0 ]
  ;;
*)
  echo "usage: $0 [precheck|parity]"; exit 64;;
esac
echo "=== D2 $MODE DONE $(date -u +%FT%TZ) out=$OUT"
