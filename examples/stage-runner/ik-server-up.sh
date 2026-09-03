#!/usr/bin/env bash
# Launch the ik_llama stage HEAD as an OpenAI-compatible server (--role server),
# driving a deployed multi-stage MTP ring. The tail must already be up in MTP mode
# (STAGE_MTP=1, STAGE_MTP_NDRAFT=<k>, --token-return <STAGE_TOKEN_RETURN>). Relays k-agnostic.
#
# All deployment-specific values are supplied via environment (no hardcoded hosts,
# paths, or model names in this file):
#   STAGE_HEAD_HOST      SSH host of the stage head (required)
#   STAGE_HEAD_USER      SSH user for the head (default: root)
#   STAGE_BIN            absolute path to llama-stage-runner on the head (required)
#   STAGE_MODEL          absolute path to the model .gguf on the head (required)
#   STAGE_TAIL           downstream tail endpoint HOST:PORT for --connect (required)
#   STAGE_TOKEN_RETURN   the head's return endpoint HOST:PORT, passed to the tail's --token-return
#   STAGE_RETURN_LISTEN  port the head listens on for tail returns (--return-listen)
#   STAGE_IL_START / STAGE_IL_END   head layer window (default 0 / 29)
#
#   usage: ik-server-up.sh [http_port] [n_ctx]      (defaults 8080 / 16384)
# curl test:
#   curl -s http://<STAGE_HEAD_HOST>:8080/v1/completions -d '{"prompt":"Hello","max_tokens":32}'
#   curl -sN http://<STAGE_HEAD_HOST>:8080/v1/chat/completions \
#        -d '{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":32}'
# Prefix-cache append is ON by default (STAGE_NO_PREFIX_CACHE to disable). Append path FIXED
# 2026-07-03 (tail re-prime + server re-seed): /v1/completions multi-turn append runs MTP
# correctly (64% accept, no churn). Chat multi-turn still full-prefills (retok roundtrip;
# correct, no cache reuse). Requires the tail on binary >= .reprime2. MTP k is a TAIL env knob only.
set -u
: "${STAGE_HEAD_HOST:?STAGE_HEAD_HOST (head SSH host) is required}"
: "${STAGE_BIN:?STAGE_BIN (path to llama-stage-runner) is required}"
: "${STAGE_MODEL:?STAGE_MODEL (path to model .gguf) is required}"
: "${STAGE_TAIL:?STAGE_TAIL (downstream tail HOST:PORT) is required}"
: "${STAGE_RETURN_LISTEN:?STAGE_RETURN_LISTEN (head return-listen port) is required}"
HEAD_USER=${STAGE_HEAD_USER:-root}
IL_START=${STAGE_IL_START:-0}
IL_END=${STAGE_IL_END:-29}
PORT=${1:-8080}
NCTX=${2:-16384}
S="ssh -o ConnectTimeout=8 ${HEAD_USER}@${STAGE_HEAD_HOST}"
${S} "pkill -9 -f 'llama-stage-runn[e]r.*role server' 2>/dev/null; sleep 1" >/dev/null 2>&1
${S} "setsid nohup env STAGE_ACTIVE=1 STAGE_IL_START=${IL_START} STAGE_IL_END=${IL_END} \
  ${STAGE_BIN} -m ${STAGE_MODEL} \
  --no-mmap --n-ctx ${NCTX} --n-ubatch 32 -ngl 999 --amb 512 --split-mode tensor \
  --role server --connect ${STAGE_TAIL} --return-listen ${STAGE_RETURN_LISTEN} --port ${PORT} \
  > /tmp/ik_server.log 2>&1 < /dev/null & disown; echo launched"
echo "waiting for :${PORT} ..."
for i in $(seq 1 40); do
  ${S} "grep -qa 'server: listening' /tmp/ik_server.log 2>/dev/null" && { echo "server up on http://${STAGE_HEAD_HOST}:${PORT}"; exit 0; }
  ${S} "grep -qaE 'load failed|Unable to auto-fit' /tmp/ik_server.log 2>/dev/null" && { echo "LOAD FAILED (see /tmp/ik_server.log)"; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for server"