#!/usr/bin/env bash
# Launch the ik_llama stage HEAD as an OpenAI-compatible server (--role server) on a GPU node,
# driving the deployed multi-stage MTP ring. The tail must already be up in MTP mode
# (STAGE_MTP=1, STAGE_MTP_NDRAFT=<k>, --token-return ${STAGE_TOKEN_RETURN}). Relays k-agnostic.
#   usage: ik-server-up.sh [http_port] [n_ctx]      (defaults 8080 / 16384)
# curl test:
#   curl -s http://${STAGE_HEAD_HOST}:8080/v1/completions -d '{"prompt":"Hello","max_tokens":32}'
#   curl -sN http://${STAGE_HEAD_HOST}:8080/v1/chat/completions \
#        -d '{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":32}'
# Prefix-cache append is ON by default (STAGE_NO_PREFIX_CACHE to disable). Append path FIXED
# 2026-07-03 (tail re-prime 55513d47 + server re-seed b0d4575): /v1/completions multi-turn append
# runs MTP correctly (64% accept, no churn). Chat multi-turn still full-prefills (retok roundtrip;
# correct, no cache reuse). Requires the tail on binary >= .reprime2. MTP k is a TAIL env knob only.
set -u
NV=${STAGE_HEAD_HOST}
PORT=${1:-8080}
NCTX=${2:-16384}
S="ssh -o ConnectTimeout=8 ${STAGE_HEAD_USER:-root}@"
${S}$NV "pkill -9 -f 'llama-stage-runn[e]r.*role server' 2>/dev/null; sleep 1" >/dev/null 2>&1
${S}$NV "setsid nohup env STAGE_ACTIVE=1 STAGE_IL_START=0 STAGE_IL_END=29 \
  ${STAGE_BIN} -m ${STAGE_MODEL} \
  --no-mmap --n-ctx $NCTX --n-ubatch 32 -ngl 999 --amb 512 --split-mode tensor \
  --role server --connect ${STAGE_TAIL} --return-listen ${STAGE_RETURN_LISTEN} --port $PORT \
  > /tmp/ik_server.log 2>&1 < /dev/null & disown; echo launched"
echo "waiting for :$PORT ..."
for i in $(seq 1 40); do
  ${S}$NV "grep -qa 'server: listening' /tmp/ik_server.log 2>/dev/null" && { echo "server up on http://$NV:$PORT"; exit 0; }
  ${S}$NV "grep -qaE 'load failed|Unable to auto-fit' /tmp/ik_server.log 2>/dev/null" && { echo "LOAD FAILED (see /tmp/ik_server.log)"; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for server"
