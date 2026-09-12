#pragma once

// Expert-tensor disaggregation, attention-side client (P1).
//
// When enabled, the MoE graph builder replaces the routed-expert FFN compute of
// the covered layers with a synchronous EXPERT_CALL/EXPERT_RET RPC to a remote
// expert-server (examples/stage-runner/expert-server.cpp). The router, the
// top-k selection, the expert weights and the shared/dense experts all stay on
// the attention side; only { hidden, topk ids, topk weights } cross the wire.
//
// Config is env-driven (the established pattern for stage plumbing in this
// tree):
//   LLAMA_EXPERTS_REMOTE        = host:port                      single server
//                               = host:port@SPEC;host:port@SPEC  one server per
//                                 layer range (P2, multi-stage placement: e.g.
//                                 stage 1's experts in the attention host's own
//                                 RAM, stage 2's on a remote high-bandwidth
//                                 node). Endpoints are separated by ';' because
//                                 a layer SPEC may itself contain ','.
//   LLAMA_EXPERTS_REMOTE_LAYERS = 3,5,10-19   absolute layer ids served remotely
//                                             (unset/empty = every MoE layer).
//                                             Only meaningful for the single-
//                                             endpoint form; with '@' specs the
//                                             per-endpoint SPEC wins and this is
//                                             ignored.
//   LLAMA_EXPERTS_REMOTE_KEEP_EXPS = 1        keep loading the exps tensors on
//                                             the attention side anyway (debug);
//                                             default is to skip them (TENSOR_SKIP)
//   LLAMA_EXPERTS_REMOTE_RETRY_MS  = 500      base reconnect backoff, doubled per
//                                             attempt and capped at 8000 ms
//
// Coverage must be a partition: two endpoints claiming the same layer is a
// fatal configuration error, as is a multi-endpoint spec with a bare address.
//
// v1 transport: one TCP connection per endpoint per process, TCP_NODELAY, one
// request in flight (the residual dependency serializes per-layer calls
// anyway). Wire or server errors tear down the connection and retry the RPC up
// to 3 times with reconnect - transient socket failures (EPIPE, brief server
// unavailability) are recovered automatically instead of aborting. Only
// persistent failures (server unreachable after retries, protocol mismatch)
// abort the process.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// ---- wire format (little-endian) -------------------------------------------
//
// EXPERT_CALL:
//   int32 magic = LLAMA_EXPERTS_MAGIC_CALL ("EXPC")
//   int32 layer            absolute layer id
//   int32 n_tok
//   int32 n_topk           experts used per token (k)
//   int32 n_embd
//   int32 ids    [n_topk * n_tok]   selected expert ids, token-major
//   f32   weights[n_topk * n_tok]   final per-expert weights (post norm/scale)
//   f32   hidden [n_embd * n_tok]   post-ffn_norm hidden states, token-major
//
// EXPERT_RET:
//   int32 magic = LLAMA_EXPERTS_MAGIC_RET ("EXPR")
//   int32 layer
//   int32 n_tok
//   int32 status           0 = ok, nonzero = server-side failure (no payload)
//   f32   out[n_embd * n_tok]       accumulated routed-expert output
//
// f32 (not f16) hidden is deliberate: it keeps the split bit-exact against the
// single-process baseline.

static const int32_t LLAMA_EXPERTS_MAGIC_CALL = 0x45585043; // "EXPC"
static const int32_t LLAMA_EXPERTS_MAGIC_RET  = 0x45585052; // "EXPR"
static const int32_t LLAMA_EXPERTS_MAGIC_CAPS = 0x45585041; // "EXPA"

// CAPS handshake (exchanged once per connection, right after TCP connect):
//   client -> server:  int32 magic = LLAMA_EXPERTS_MAGIC_CAPS, int32 version
//   server -> client:  int32 magic, int32 version, int32 n_served, int32 n_embd,
//                      int32 served_layer[n_served], int32 backend_id, int32 n_threads
// version is the wire-format version of this feature (1). A mismatch aborts the
// connection before any EXPERT_CALL is sent. The client verifies version and
// n_embd, and verifies that its assigned layer set is a subset of served_layer.
static const int32_t LLAMA_EXPERTS_VERSION = 1;

struct llama_experts_remote_endpoint {
    std::string host;
    int         port = 0;
};

struct llama_experts_remote_cfg {
    bool enabled   = false;
    bool keep_exps = false;             // load exps tensors on the attention side anyway

    std::vector<llama_experts_remote_endpoint> endpoints;

    // absolute layer id -> index into endpoints; -1 = not covered (compute locally)
    std::vector<int32_t> layer_ep;

    // endpoint that covers every MoE layer without an explicit list (the legacy
    // single-server form, i.e. no '@SPEC' and no LLAMA_EXPERTS_REMOTE_LAYERS); -1 = none
    int32_t default_ep = -1;

    int endpoint_for(int il) const {
        if (!enabled || il < 0) {
            return -1;
        }
        if ((size_t) il < layer_ep.size() && layer_ep[il] >= 0) {
            return layer_ep[il];
        }
        return default_ep;
    }

    bool layer_covered(int il) const {
        return endpoint_for(il) >= 0;
    }
};

// parsed once from the environment (first call), then cached
const llama_experts_remote_cfg & llama_experts_remote_get_cfg();

// ggml_map_custom3 callback: a = hidden [n_embd, n_tok] f32,
// b = selected expert ids [n_topk, n_tok] i32, c = weights [1, n_topk, n_tok] f32,
// userdata = (intptr_t) absolute layer id. Performs the blocking RPC.
void llama_experts_remote_custom_cb(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith, int nth, void * userdata);
