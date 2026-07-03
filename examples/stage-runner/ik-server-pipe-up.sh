#!/usr/bin/env bash
# Launch the ik_llama stage HEAD as an OpenAI server in MULTI-SLOT PIPELINED mode (--slots N>1),
# driving the deployed multi-stage MTP ring. Up to N independent requests keep one wave each in
# flight so different ring stages process different requests simultaneously (pipeline parallelism).
# The synchronous single-slot server (ik-server-up.sh) processes one wave end-to-end per token.
#
#   usage: ik-server-pipe-up.sh [slots] [n_seq_max] [http_port] [n_ctx]   (defaults 4 8 8080 16384)
#
# REQUIREMENTS (already true on the live GLM ring unless reconfigured):
#   * Tail launched with STAGE_N_REAL >= slots  (q4_tail: STAGE_N_REAL=8) else seqs>=g_n_real drop.
#   * Every stage --n-seq-max >= slots so each seq's ctx partition (n_ctx/n_seq_max) holds prompt+gen.
#     The GLM relays currently run the DEFAULT n_seq_max=64 -> only 256 positions/seq. Until they are
#     relaunched with --n-seq-max 8, keep (prompt+max_tokens) < 256 per request. The tail is already
#     at --n-seq-max 8 (2048/seq).
# curl (fire 4 concurrent):
#   for i in 1 2 3 4; do curl -s :8080/v1/completions -d '{"prompt":"Count: 1 2 3","max_tokens":40}' & done; wait
set -u
NV=${STAGE_HEAD_HOST}
SLOTS=${1:-4}
NSEQ=${2:-8}
PORT=${3:-8080}
NCTX=${4:-16384}
S="ssh -o ConnectTimeout=8 ${STAGE_HEAD_USER:-root}@"
${S}$NV "pkill -9 -f 'llama-stage-runn[e]r.*role server' 2>/dev/null; sleep 1" >/dev/null 2>&1
${S}$NV "setsid nohup env STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=29 \
  ${STAGE_BIN} -m ${STAGE_MODEL} \
  --no-mmap --n-ctx $NCTX --n-seq-max $NSEQ --n-ubatch 32 -ngl 999 --amb 512 --split-mode tensor \
  --role server --slots $SLOTS --connect ${STAGE_TAIL} --return-listen ${STAGE_RETURN_LISTEN} --port $PORT \
  > /tmp/ik_server.log 2>&1 < /dev/null & disown; echo launched"
echo "waiting for :$PORT (pipelined, slots=$SLOTS n_seq_max=$NSEQ) ..."
for i in $(seq 1 40); do
  ${S}$NV "grep -qa 'server\[pipe\]: listening' /tmp/ik_server.log 2>/dev/null" && { echo "server up on http://$NV:$PORT"; exit 0; }
  ${S}$NV "grep -qaE 'load failed|Unable to auto-fit|connect FAILED' /tmp/ik_server.log 2>/dev/null" && { echo "LAUNCH FAILED (see /tmp/ik_server.log)"; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for server"; exit 1
