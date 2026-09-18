#!/bin/bash
# D2 Inkling parity runner — ben 2026-09-17/18. Counterpart to run-d0-oracle.ben.sh.
#
# Modes:
#   precheck   Internal build on the 77 GB IQ2_M quant, ppl with default -fa vs -fa 0, diffed.
#              SETTLED 2026-09-18 on the real model: the paths DIFFER (banded 72.6183 vs masked
#              77.0969), so the recorded D0 oracle is D3's comparand, NOT D2's. Kept for reruns.
#   refgreedy  Record the MASKED-PATH greedy reference: noah's internal build, llama-server -fa 0,
#              the D0 prompt set and request body. D2 executes the masked path, so THIS -- not the
#              recorded banded greedy-64x8.json -- is what D2's tokens must match. Writes $REFM/.
#   parity     This tree's D2 build on Q4_K_M, -fa 0: greedy token-for-token vs $REFM, and the four
#              per-chunk ppl values vs the masked-path capture from the 2026-09-18 01:57 window.
#
# PRECONDITION (booked window): resident model servers quieted; MemAvailable >= 160 GiB.
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
ORACLE=/fast/build/agents/noah/p1/d0-oracle-20260915T2024    # banded-path capture (D3's gate);
                                                             # used here ONLY for the prompt set
MASKED_PPL=$D2/window-20260918T0156/ppl-fa-off.log            # reference, masked path, real Q4_K_M
REFM=$D2/ref-masked                                          # masked-path greedy reference (refgreedy)
OUT=$D2/d2-$MODE-$(date -u +%Y%m%dT%H%M)
THREADS=20
MEM_FLOOR_GIB=160
MEM_ABORT_KB=$((12*1024*1024))

# Inkling has no banded FA in this tree, so build_inkling asserts unless FA is off. The bias is
# per-head and ggml_flash_attn_ext never materialises kq, so there is nothing to add it to.
FA_OFF="-fa off"

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

chunks() { grep -oE '\[1\][0-9.]+,\[2\][0-9.]+,\[3\][0-9.]+,\[4\][0-9.]+,' "$1" | tail -1; }

ppl_run() {  # ppl_run <binary> <model> <extra-args> <logfile>
  "$1" -m "$2" -ngl 0 -t $THREADS -f "$WIKI" -c 2048 -b 2048 --chunks 4 $3 > "$4" 2>&1
  echo "  ppl rc=$? -> $4"
  echo "  chunks: $(chunks "$4")"
}

# run_greedy <server-binary> <port> <out.json> [compare.json]
# Drives the D0 prompt set (read from the banded recording, prompts only) through a server with
# the D0 request body. With compare.json, checks tokens position-for-position and returns 1 on any
# mismatch. Provenance is the server binary hash: a parity number without provenance is not evidence.
run_greedy() {
  local bin=$1 port=$2 out=$3 cmp=${4:-}
  echo "--- greedy: server=$(sha256sum "$bin" | cut -c1-12) port=$port -> $out"
  "$bin" -m "$M_Q4" -ngl 0 -t $THREADS -c 4096 -np 1 --jinja $FA_OFF \
      --host 127.0.0.1 --port $port > "$out.server.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 360); do
    curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && break
    kill -0 $SP 2>/dev/null || { echo "SERVER-DIED"; tail -30 "$out.server.log"; return 3; }
    sleep 5
  done
  curl -sf http://127.0.0.1:$port/health >/dev/null || { echo "SERVER-NOT-HEALTHY"; return 3; }
  memlog server-healthy
  python3 - "$port" "$out" "$ORACLE/greedy-64x8.json" "$cmp" <<'PY'
import json,sys,urllib.request,hashlib
port,out,prompts_from,cmp=sys.argv[1:5]
prompts=[r["prompt"] for r in json.load(open(prompts_from))]
ref=json.load(open(cmp)) if cmp else None
res=[];fails=0
for i,p in enumerate(prompts):
    body={"prompt":p,"n_predict":64,"temperature":0,"top_k":1,"seed":1,
          "cache_prompt":False,"return_tokens":True,"n_probs":10}
    r=json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}),timeout=1800))
    got=r.get("tokens")
    rec={"i":i,"prompt":p,"tokens":got,"content":r.get("content"),
         "completion_probabilities":r.get("completion_probabilities"),"timings":r.get("timings")}
    sha=hashlib.sha256(json.dumps(got).encode()).hexdigest()[:16]
    if ref is not None:
        want=ref[i]["tokens"]; ok=(got==want); rec["match"]=ok
        if ok: print(f"prompt {i}: MATCH ({len(got)} tokens) sha={sha}",flush=True)
        else:
            fails+=1; n=min(len(got or []),len(want or []))
            first=next((k for k in range(n) if got[k]!=want[k]), n)
            print(f"prompt {i}: MISMATCH at token {first} (got {got[first:first+4] if got else None} want {want[first:first+4]}) sha={sha}",flush=True)
    else:
        print(f"prompt {i}: {len(got or [])} tokens, {round((r.get('timings') or {}).get('predicted_per_second',0),2)} tok/s, sha={sha}",flush=True)
    res.append(rec)
json.dump(res,open(out,"w"),indent=1)
if ref is not None: print(f"GREEDY: {len(prompts)-fails}/{len(prompts)} prompts token-for-token identical")
sys.exit(1 if fails else 0)
PY
  local rc=$?
  kill -TERM $SP; wait $SP 2>/dev/null; SP=""; echo "$(date -u +%T) server stopped"
  memlog after-greedy
  return $rc
}

case "$MODE" in
# ---------------------------------------------------------------- precheck
precheck)
  echo "=== D2 PRECHECK (banded-FA question) $(date -u +%FT%TZ) host=$(hostname)"
  precond "$M_IQ2" "$WIKI" "$NOAH_B/llama-perplexity"
  watchdog
  echo "--- run A: default -fa"; ppl_run "$NOAH_B/llama-perplexity" "$M_IQ2" "" "$OUT/ppl-fa-auto.log"
  echo "--- run B: -fa 0";       ppl_run "$NOAH_B/llama-perplexity" "$M_IQ2" "-fa 0" "$OUT/ppl-fa-off.log"
  A=$(chunks "$OUT/ppl-fa-auto.log"); B2=$(chunks "$OUT/ppl-fa-off.log")
  echo "fa auto: $A"; echo "fa 0   : $B2"
  [ "$A" = "$B2" ] && echo "VERDICT: IDENTICAL" || echo "VERDICT: DIFFER -> recorded oracle is not D2's comparand"
  ;;
# ---------------------------------------------------------------- refgreedy
refgreedy)
  echo "=== D2 REFGREEDY (masked-path greedy reference) $(date -u +%FT%TZ) host=$(hostname)"
  echo "    binary = noah's internal build (has the banded op; forced OFF with $FA_OFF)"
  precond "$M_Q4" "$NOAH_B/llama-server" "$ORACLE/greedy-64x8.json"
  watchdog
  mkdir -p "$REFM"
  run_greedy "$NOAH_B/llama-server" 18098 "$REFM/greedy-64x8-masked.json"; rc=$?
  echo "refgreedy rc=$rc"
  ( cd "$REFM" && sha256sum greedy-64x8-masked.json > SHA256SUMS ) && cat "$REFM/SHA256SUMS"
  [ $rc -eq 0 ]
  ;;
# ---------------------------------------------------------------- parity
parity)
  # Provenance: the staged tree has no .git (it is rsynced without it), so fall back to hashing
  # the binaries actually being run.
  TREE=$(git -C $D2 rev-parse --short HEAD 2>/dev/null)
  [ -n "$TREE" ] || TREE="nogit:server=$(sha256sum $B/llama-server | cut -c1-12),ppl=$(sha256sum $B/llama-perplexity | cut -c1-12)"
  echo "=== D2 PARITY $(date -u +%FT%TZ) host=$(hostname) tree=$TREE"
  echo "    model=$M_Q4  fa=OFF(masked)  threads=$THREADS"
  echo "    greedy comparand = $REFM/greedy-64x8-masked.json  (masked path, NOT the banded oracle)"
  echo "    ppl    comparand = $MASKED_PPL"
  precond "$M_Q4" "$WIKI" "$B/llama-server" "$B/llama-perplexity" \
          "$REFM/greedy-64x8-masked.json" "$MASKED_PPL" "$ORACLE/greedy-64x8.json"
  watchdog

  # ---- (1) greedy vs the masked-path reference (the oracle drove this via llama-server) ----
  run_greedy "$B/llama-server" 18099 "$OUT/d2-greedy-64x8.json" "$REFM/greedy-64x8-masked.json"; grc=$?
  echo "greedy compare rc=$grc"

  # ---- (2) perplexity vs the masked-path four chunk values ----
  ppl_run "$B/llama-perplexity" "$M_Q4" "$FA_OFF --kl-divergence-base $OUT/d2-kld-4x2048.bin" "$OUT/ppl.log"
  memlog after-ppl
  got=$(chunks "$OUT/ppl.log"); want=$(chunks "$MASKED_PPL")
  echo "ppl reference (masked): $want"
  echo "ppl d2                : $got"
  prc=0; [ "$got" = "$want" ] || prc=1

  ( cd "$OUT" && sha256sum d2-greedy-64x8.json ppl.log > SHA256SUMS ) 2>/dev/null
  echo
  if [ $grc -eq 0 ] && [ $prc -eq 0 ]; then
    echo "=== D2 PARITY PASS (greedy token-for-token AND 4/4 ppl chunks vs the masked-path reference) out=$OUT"
  else
    echo "=== D2 PARITY FAIL (greedy rc=$grc, ppl rc=$prc) out=$OUT"
    echo "    Both comparands are the reference's MASKED path, matching what D2 runs."
  fi
  [ $grc -eq 0 ] && [ $prc -eq 0 ]
  ;;
*)
  echo "usage: $0 [precheck|refgreedy|parity]"; exit 64;;
esac
echo "=== D2 $MODE DONE $(date -u +%FT%TZ) out=$OUT"
