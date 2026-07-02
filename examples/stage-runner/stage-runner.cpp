// stage-runner: run a llama.cpp model as a PIPELINE STAGE (TCP hidden-state handoff).
//
// One model is split across instances. Each executes only its layer window
// [STAGE_IL_START, STAGE_IL_END) (env, consumed by the patched graph builders).
// Hidden states are handed off as raw tensors over TCP — point-to-point,
// vendor-agnostic, no ggml-rpc control round-trips. Each stage keeps its own KV
// locally.
//
// This is the ik_llama port: TCP-only, NON-pipelined (depth==slots), v1.
//   DROPPED for v1 (vs the mainline stage-keepalive driver): all MTP /
//   NextN self-speculation, the pipelined depth-D variants (run_*_pipe),
//   middle / router / relay2 / bench roles, filler / keepalive logic,
//   RDMA transport, and STAGE_FA handling. See the dropped-for-v1 comments below.
//
// Modes:
//  FILE (single forward, correctness harness):
//    --prompt P | --in F ; --out F (env STAGE_EMIT=hidden) ; --last (argmax)
//  RING (persistent, KV-cached, multi-slot decode over TCP):
//    --role head : tokenize C copies of the prompt (slots 0..C-1), embed+run
//                  window, emit hidden; recv sampled tokens back, continue.
//    --role relay: recv hidden from upstream -> run window -> forward downstream;
//                  relay sampled tokens back upstream. --listen PORT / --connect HOST:PORT
//    --role tail : recv hidden, run window + final norm/head, sample per slot,
//                  send tokens back. --listen PORT
//    --slots C   (head) number of concurrent sequences ; --max-tokens N (tail)
//
//  env: STAGE_ACTIVE=1, STAGE_IL_START, STAGE_IL_END, STAGE_EMIT(hidden|logits)

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
// NOTE: the mainline driver included "../../src/llama-ext.h" for the custom
// llama_*_pre_norm staging API. ik_llama exposes the public embeddings API
// instead (llama_get_embeddings_ith / llama_set_embeddings), so that include is
// dropped here and every *_pre_norm call is rewired below.
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cmath>

static const int32_t STAGE_MAGIC = 0x53544732; // "STG2" (v2: per-row tagged)

// ---- hidden-state blob: n_rows rows, each tagged (seq,pos), n_embd floats ----
struct hidden_blob {
    int32_t n_rows = 0;
    int32_t n_embd = 0;
    std::vector<int32_t> seq;   // n_rows
    std::vector<int32_t> pos;   // n_rows
    std::vector<float>   data;  // n_rows * n_embd
    void resize(int rows, int embd) {
        n_rows = rows; n_embd = embd;
        seq.resize(rows); pos.resize(rows); data.resize((size_t) rows * embd);
    }
};

// ---- socket helpers ----
static bool send_all(int fd, const void * b, size_t n) {
    const char * p = (const char *) b;
    while (n) { ssize_t k = send(fd, p, n, 0); if (k <= 0) return false; p += k; n -= (size_t) k; }
    return true;
}
static bool recv_all(int fd, void * b, size_t n) {
    char * p = (char *) b;
    while (n) { ssize_t k = recv(fd, p, n, 0); if (k <= 0) return false; p += k; n -= (size_t) k; }
    return true;
}
static bool send_hidden(int fd, const hidden_blob & h) {
    int32_t hdr[3] = { STAGE_MAGIC, h.n_rows, h.n_embd };
    return send_all(fd, hdr, sizeof(hdr)) &&
           send_all(fd, h.seq.data(),  h.n_rows * sizeof(int32_t)) &&
           send_all(fd, h.pos.data(),  h.n_rows * sizeof(int32_t)) &&
           send_all(fd, h.data.data(), h.data.size() * sizeof(float));
}
static bool recv_hidden(int fd, hidden_blob & h) {
    int32_t hdr[3];
    if (!recv_all(fd, hdr, sizeof(hdr)) || hdr[0] != STAGE_MAGIC) return false;
    h.resize(hdr[1], hdr[2]);
    return recv_all(fd, h.seq.data(),  h.n_rows * sizeof(int32_t)) &&
           recv_all(fd, h.pos.data(),  h.n_rows * sizeof(int32_t)) &&
           recv_all(fd, h.data.data(), h.data.size() * sizeof(float));
}
// back edge: C sampled tokens (one per slot) + a global eog flag
static bool send_tokens(int fd, const std::vector<int32_t> & toks, int32_t eog) {
    int32_t n = (int32_t) toks.size();
    int32_t h[2] = { n, eog };
    return send_all(fd, h, sizeof(h)) && send_all(fd, toks.data(), n * sizeof(int32_t));
}
static bool recv_tokens(int fd, std::vector<int32_t> & toks, int32_t & eog) {
    int32_t h[2];
    if (!recv_all(fd, h, sizeof(h))) return false;
    toks.resize(h[0]); eog = h[1];
    return recv_all(fd, toks.data(), h[0] * sizeof(int32_t));
}

// MTP back-edge (wire-compatible with the mainline tail's run_tail_pipe_mtp): one record per
// returned wave on the DIRECT tail->head socket. The tail owns verify/draft and dictates the
// head's next verify wave = `issue` = [conf, d_1..d_k] (k+1 rows at p_base..p_base+k).
struct mtp_msg {
    int32_t eog = 0;
    int32_t n_new = 0;             // # newly-confirmed real tokens this wave (0 = filler/dummy)
    int32_t p_base = 0;            // base pos of the NEXT wave
    std::vector<int32_t> issue;    // next wave tokens [conf, d_1..d_k]; empty for filler/eog
    std::vector<int32_t> out;      // the n_new confirmed tokens to emit, in order
};
static bool recv_mtp_msg(int fd, mtp_msg & m) {
    int32_t hdr[4];
    if (!recv_all(fd, hdr, sizeof(hdr))) return false;
    m.eog = hdr[0]; m.n_new = hdr[1]; m.p_base = hdr[2];
    m.issue.resize(hdr[3] > 0 ? hdr[3] : 0);
    m.out.resize(m.n_new > 0 ? m.n_new : 0);
    if (hdr[3] > 0 && !recv_all(fd, m.issue.data(), (size_t) hdr[3] * sizeof(int32_t))) return false;
    if (m.n_new > 0 && !recv_all(fd, m.out.data(), (size_t) m.n_new * sizeof(int32_t))) return false;
    return true;
}
// DROPPED for v1: mtp_msg / send_mtp_msg / recv_mtp_msg (MTP back-edge protocol).

static bool write_file(const char * path, const hidden_blob & h) {
    FILE * f = fopen(path, "wb"); if (!f) return false;
    int32_t hdr[3] = { STAGE_MAGIC, h.n_rows, h.n_embd };
    bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
              fwrite(h.seq.data(), sizeof(int32_t), h.n_rows, f) == (size_t) h.n_rows &&
              fwrite(h.pos.data(), sizeof(int32_t), h.n_rows, f) == (size_t) h.n_rows &&
              fwrite(h.data.data(), sizeof(float), h.data.size(), f) == h.data.size();
    fclose(f); return ok;
}
static bool read_file(const char * path, hidden_blob & h) {
    FILE * f = fopen(path, "rb"); if (!f) { fprintf(stderr,"stage: open %s failed\n",path); return false; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != STAGE_MAGIC) { fclose(f); return false; }
    h.resize(hdr[1], hdr[2]);
    bool ok = fread(h.seq.data(), sizeof(int32_t), h.n_rows, f) == (size_t) h.n_rows &&
              fread(h.pos.data(), sizeof(int32_t), h.n_rows, f) == (size_t) h.n_rows &&
              fread(h.data.data(), sizeof(float), h.data.size(), f) == h.data.size();
    fclose(f); return ok;
}
// caps HELLO interop with the mainline-fork stage_conn transport: every mainline stage
// exchanges a 24-byte RDMA-caps blob at connect/accept (connector sends first). An all-zero
// blob means "no RDMA" (qpn==0 -> clean TCP fallback on the peer), so this pre-RDMA build
// just mirrors the exchange with zeros. Without it a hardened mainline peer drops the conn
// ("caps_hello failed") or reads the first wave header as caps (byte desync).
static const size_t STAGE_CAPS_SIZE = 24;
static bool caps_hello_shim(int fd, bool client_first) {
    uint8_t zeros[STAGE_CAPS_SIZE] = {0}, peer[STAGE_CAPS_SIZE];
    if (client_first) return send_all(fd, zeros, sizeof(zeros)) && recv_all(fd, peer, sizeof(peer));
    return recv_all(fd, peer, sizeof(peer)) && send_all(fd, zeros, sizeof(zeros));
}
static int tcp_listen_accept(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
    if (bind(s, (sockaddr *) &a, sizeof(a)) < 0) { perror("bind"); return -1; }
    if (listen(s, 4) < 0) { perror("listen"); return -1; }
    fprintf(stderr, "stage: listening on :%d\n", port);
    for (;;) {
        int c = accept(s, nullptr, nullptr);
        if (c < 0) { close(s); return -1; }
        int f = 1; setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f));
        if (!caps_hello_shim(c, /*client_first=*/false)) {   // dead backlog conn: drop, re-accept
            fprintf(stderr, "stage: accept caps_hello failed -> re-accept\n");
            close(c); continue;
        }
        close(s);
        return c;
    }
}
static int tcp_connect(const std::string & host, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    for (int t = 0; t < 100; ++t) {
        if (connect(s, (sockaddr *) &a, sizeof(a)) == 0) {
            int f = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f));
            if (!caps_hello_shim(s, /*client_first=*/true)) {   // peer dropped mid-hello: retry fresh
                fprintf(stderr, "stage: connect caps_hello failed -> retry\n");
                close(s);
                s = socket(AF_INET, SOCK_STREAM, 0);
                usleep(100000); continue;
            }
            return s;
        }
        usleep(100000);
    }
    perror("connect"); return -1;
}

// ---- model ----
// STAGE_DUMP=<substr>: gather (via ggml_backend_tensor_get) every F32 tensor whose
// name contains <substr> and write it to /work/dump/<name>.bin (header: 4x int64 ne,
// then raw floats). Correctness-debug only.
// ik_llama's ggml_backend_sched_eval_callback returns int (mainline returned bool).
static int stage_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const char * pat = (const char *) ud;
    if (!pat || !pat[0]) return false;
    // pat is a comma-separated list of substrings; match if the name contains ANY token.
    auto matches = [&](const char * name) -> bool {
        std::string list(pat); size_t a = 0;
        while (a < list.size()) {
            size_t c = list.find(',', a); std::string tok = list.substr(a, c==std::string::npos?std::string::npos:c-a);
            if (!tok.empty() && strstr(name, tok.c_str())) return true;
            if (c==std::string::npos) break; a = c+1;
        }
        return false;
    };
    if (ask) return matches(t->name);
    if (!matches(t->name)) return true;
    if (t->type != GGML_TYPE_F32) { fprintf(stderr, "[DUMP] skip %s (type=%d)\n", t->name, (int) t->type); return true; }
    if (!ggml_is_contiguous(t)) { fprintf(stderr, "[DUMP] skip %s (noncontig)\n", t->name); return true; }
    const size_t nb = ggml_nbytes(t);
    std::vector<char> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    static bool mkd = false; if (!mkd) { mkdir("/work/dump", 0777); mkd = true; }
    std::string fn = "/work/dump/";
    for (const char * c = t->name; *c; ++c) fn += (*c=='/'||*c==' '||*c=='(' ||*c==')') ? '_' : *c;
    fn += ".bin";
    FILE * f = fopen(fn.c_str(), "wb");
    if (f) {
        int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        fwrite(ne, sizeof(ne), 1, f); fwrite(buf.data(), 1, nb, f); fclose(f);
        fprintf(stderr, "[DUMP] %s [%lld,%lld,%lld,%lld]\n", t->name,
                (long long)ne[0],(long long)ne[1],(long long)ne[2],(long long)ne[3]);
    }
    return true;
}

struct model_bundle {
    llama_model * model = nullptr; llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr; int n_embd = 0, n_vocab = 0;
    int n_ubatch = 0;   // physical ubatch cap (= cp.n_ubatch); run_hidden chunks the emit to this so embeddings extraction stays single-ubatch (multi-slot fix)
};
static int g_amb = 0;   // --amb: ik attn_max_batch (caps the attention compute scratch; mandatory at long ctx)
static bool load(model_bundle & b, const std::string & path, int ngl, int n_ctx, bool use_mmap,
                 const std::vector<float> & tsplit, int split_mode, int n_ubatch,
                 const std::vector<llama_model_tensor_buft_override> & buft_ovr) {
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl; mp.use_mmap = use_mmap;
    // ik's llm_load_tensors VRAM planner sizes per-device compute buffers from MODEL params
    // (max_ctx_size, n_seq_max, n_ubatch, amb) — defaults left unset made it demand 128 GiB/device
    // (n_seq_max=64 x 16K ctx) and misplace every layer. Feed it the real run shape.
    mp.max_ctx_size = (uint32_t) n_ctx;
    mp.n_seq_max    = 1;                     // single-slot stage; raise with --slots when multi-seq lands
    mp.n_ubatch     = (n_ubatch > 0) ? n_ubatch : (n_ctx < 2048 ? n_ctx : 2048);
    if (g_amb > 0) mp.amb = g_amb;
    mp.fit = true;   // ik auto-fit: planner adds per-layer expert-CPU overrides until the model fits
                     // (fleet-preferred over manual -ot; no-op when everything fits on GPU)
    if (split_mode >= 0) mp.split_mode = (enum llama_split_mode) split_mode;   // 1=layer(pipeline), 2=row/attn(TP)
    // ---- ik_llama fast-path model flags (mirror common.cpp's mparams.* mapping) ----
    // -rtr / run-time tensor repack: model param `repack_tensors` (confirmed include/llama.h).
    if (getenv("STAGE_RTR")) mp.repack_tensors = true;
    // -mla / MLA attention level for deepseek2/minimax: model param `mla` (int32_t).
    // The context-side mla_attn is set in cp below; both must agree (common.cpp sets both).
    // MLA level: model param mp.mla (builds the wk_b/wv_b absorb tensors at load) and the
    // context param cp.mla_attn (selects the graph path) MUST AGREE — common.cpp sets both
    // from one value. Default 3 (matches common.h). A mismatch -> null wk_b -> segfault.
    const int stage_mla = getenv("STAGE_MLA") ? atoi(getenv("STAGE_MLA")) : 3;
    mp.mla = stage_mla;
    // A stage's layer window may fall entirely in one GPU's slice under the default
    // layer-split (which maps by full-model layer index). --tensor-split forces the
    // window to spread across the local GPUs.
    static float ts_arr[128] = {0};   // llama may read up to llama_max_devices(); pad with zeros
    if (!tsplit.empty()) {
        for (size_t i = 0; i < tsplit.size() && i < 128; ++i) ts_arr[i] = tsplit[i];
        mp.tensor_split = ts_arr;
    }
    // CPU/alt-buffer tensor offload (--override-tensor / --cpu-moe): redirect matched tensors
    // (e.g. an in-window layer's MoE experts) off-GPU to free VRAM so a middle stage can absorb
    // more layers shed from the slow tail. NULL-terminated list ({nullptr,nullptr} last element).
    if (!buft_ovr.empty()) mp.tensor_buft_overrides = buft_ovr.data();
    b.model = llama_model_load_from_file(path.c_str(), mp);
    if (!b.model) { fprintf(stderr, "stage: load failed\n"); return false; }
    b.vocab = llama_model_get_vocab(b.model);
    b.n_embd = llama_model_n_embd(b.model);
    b.n_vocab = llama_vocab_n_tokens(b.vocab);
    llama_context_params cp = llama_context_default_params();
    // n_ubatch override: A770 (Intel ARC) Vulkan MUL_MAT_ID (MoE expert matmul) for
    // i-quants HANGS the compute engine when the physical ubatch token count > 8.
    // Capping n_ubatch to 8 on A770 forces every prefill ubatch through the vec path.
    cp.n_ctx = n_ctx; cp.n_batch = (n_ctx < 2048 ? n_ctx : 2048); cp.n_ubatch = (n_ubatch > 0) ? n_ubatch : cp.n_batch;   // mainline parity: unbounded n_batch sizes compute buffers for n_ctx-token batches (131 GiB/device at 16K)
    cp.n_seq_max = 64; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (g_amb > 0) cp.attn_max_batch = g_amb;
    // NOTE: mainline's cp.no_perf does not exist on ik_llama's llama_context_params; dropped.
    // Emit stages need embeddings on so llama_get_embeddings_ith() returns the per-row
    // hidden state (the staging handoff). Pooling stays NONE so rows are returned in
    // batch order, one per output row (not pooled by sequence).
    // CRITICAL: per-role. An EMIT stage (STAGE_EMIT=hidden) sets embeddings=true and the
    // graph skips output_norm+lm_head -> res=nullptr, embd extracted. A TAIL (no STAGE_EMIT)
    // MUST keep embeddings=false so the graph runs lm_head and logits are extracted for sampling.
    { const char * em = getenv("STAGE_EMIT"); cp.embeddings = (em && strcmp(em, "hidden") == 0); }
    // ---- ik_llama fast-path context flags (mirror common.cpp's cparams.* mapping) ----
    // -fmoe / fused MoE up+gate op: context param `fused_moe_up_gate` (bool). Default ON
    // here (common.cpp defaults it true; -no-fmoe disables); STAGE_NO_FMOE turns it off.
    cp.fused_moe_up_gate = getenv("STAGE_NO_FMOE") ? false : true;
    // -mla / MLA attention level: context param `mla_attn` (int). Keep in sync with mp.mla.
    cp.mla_attn = stage_mla;   // MUST match mp.mla so wk_b/wv_b exist (else build_deepseek2:591 null deref)
    { const char * th = getenv("STAGE_THREADS"); if (th) { int n = atoi(th); if (n > 0) { cp.n_threads = n; cp.n_threads_batch = n; } } }
    // DROPPED for v1: STAGE_FA handling (flash-attn force on/off); left unset -> llama AUTO.
    if (const char * dp = getenv("STAGE_DUMP")) { cp.cb_eval = stage_eval_cb; cp.cb_eval_user_data = (void *) dp; }
    b.ctx = llama_init_from_model(b.model, cp);
    if (!b.ctx) { fprintf(stderr, "stage: ctx failed\n"); return false; }
    b.n_ubatch = (int) cp.n_ubatch;
    return true;
}

// DROPPED for v1: mtp_kv_trim (MTP KV rollback before re-decoding a rejected draft).

// ---- per-stage compute timing (STAGE_TIMING=1): windowed avg llama_decode time ----
// Each stage logs its own avg decode (compute) time; summing across stages vs the
// end-to-end per-token latency isolates compute from node-to-node traversal.
static void timing_tick(long dt_ns) {
    static bool on = getenv("STAGE_TIMING") != nullptr;
    if (!on) return;
    static long w_ns = 0, w_n = 0;
    w_ns += dt_ns; w_n++;
    if (w_n >= 50) {
        fprintf(stderr, "stage[compute]: avg %.2f ms/decode over last %ld\n", w_ns/1e6/(double)w_n, w_n);
        w_ns = 0; w_n = 0;
    }
}

// Decode a TOKEN batch (head): rows tagged by (seq,pos). Extract hidden per row.
static bool run_tokens(model_bundle & b, const std::vector<int32_t> & tok,
                       const std::vector<int32_t> & seq, const std::vector<int32_t> & pos,
                       hidden_blob & out) {
    int n = (int) tok.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tok[i]; batch.pos[i] = pos[i];
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq[i]; batch.logits[i] = 1;
    }
    auto _t0 = std::chrono::steady_clock::now();
    bool ok = llama_decode(b.ctx, batch) == 0;
    timing_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t0).count());
    if (ok) {
        out.resize(n, b.n_embd); out.seq = seq; out.pos = pos;
        for (int i = 0; i < n; ++i) {
            const float * h = llama_get_embeddings_ith(b.ctx, i);
            if (!h) { ok = false; break; }
            memcpy(out.data.data() + (size_t) i * b.n_embd, h, b.n_embd * sizeof(float));
        }
    }
    llama_batch_free(batch);
    return ok;
}

// Decode an EMBD batch (middle/tail): rows tagged by (seq,pos).
// emit -> fill out with hidden; else leave logits in ctx for sampling.
static bool run_hidden(model_bundle & b, const hidden_blob & in, bool emit, hidden_blob & out) {
    int n = in.n_rows;
    if (n <= 0) { if (emit) out.resize(0, b.n_embd); return true; }
    // Chunk the EMIT path to <= n_ubatch rows per llama_decode. A single llama_decode of
    // n > n_ubatch rows splits internally into ubatches, but the hidden-state extraction
    // (llama_get_embeddings_ith) is only valid for ONE ubatch (a fill path
    // overwrites at offset 0), so a multi-ubatch decode keeps only the LAST ubatch's rows
    // -> multi-slot collapse (head --slots N -> downstream sees 1). Decoding <= n_ubatch
    // rows per call keeps every decode single-ubatch (emit correct); we accumulate across
    // chunks here. Bonus: keeps MUL_MAT_ID <= n_ubatch(=8) tok on A770 -> no i-quant hang.
    // Only chunk when emit: the tail decodes emit=false and samples via argmax_ith (logits
    // are valid for one decode only), and the tail (max, gfx1151) uses large n_ubatch anyway.
    const int chunk = (emit && b.n_ubatch > 0 && b.n_ubatch < n) ? b.n_ubatch : n;
    if (emit) { out.resize(n, b.n_embd); out.seq = in.seq; out.pos = in.pos; }
    for (int off = 0; off < n; off += chunk) {
        const int m = (n - off < chunk) ? (n - off) : chunk;
        llama_batch batch = llama_batch_init(m, b.n_embd, 1);
        batch.n_tokens = m;
        for (int i = 0; i < m; ++i) {
            memcpy(batch.embd + (size_t) i * b.n_embd, in.data.data() + (size_t) (off + i) * b.n_embd, b.n_embd * sizeof(float));
            batch.pos[i] = in.pos[off + i];
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = in.seq[off + i]; batch.logits[i] = 1;
        }
        auto _t0 = std::chrono::steady_clock::now();
        bool ok = llama_decode(b.ctx, batch) == 0;
        timing_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t0).count());
        if (ok && emit) {
            for (int i = 0; i < m; ++i) {
                const float * h = llama_get_embeddings_ith(b.ctx, i);
                if (!h) { ok = false; break; }
                memcpy(out.data.data() + (size_t) (off + i) * b.n_embd, h, b.n_embd * sizeof(float));
            }
        }
        llama_batch_free(batch);
        if (!ok) return false;
    }
    return true;
}
static int argmax_ith(model_bundle & b, int i) {
    const float * lg = llama_get_logits_ith(b.ctx, i);
    if (!lg) return -1;
    int best = 0; for (int v = 1; v < b.n_vocab; ++v) if (lg[v] > lg[best]) best = v;
    return best;
}
// Kept from the reference (only the dropped pipelined/MTP head used it). Marked
// unused so a -Werror=unused-function build still compiles in this v1.
__attribute__((unused))
static void print_piece(model_bundle & b, int tok, int slot) {
    char buf[256]; int n = llama_token_to_piece_vocab(b.vocab, tok, buf, sizeof(buf), 0, true);
    if (n > 0) { printf("[s%d]%.*s\n", slot, n, buf); fflush(stdout); }
}

// DROPPED for v1: run_middle (replica) / run_router (hub) roles.
// DROPPED for v1: run_relay_pipe / run_tail_pipe / run_*_pipe_mtp (pipelined depth-D ring).
// DROPPED for v1: make_mtp_ctx / mtp_draft / mtp_draft_h / build_chain (NextN self-speculation).

int main(int argc, char ** argv) {
    std::string model_path, prompt, in_path, out_path, role, connect_to;
    int ngl = 999, n_ctx = 4096, listen_port = 0, max_tokens = 64, slots = 1, n_ubatch = 0, amb = 0, return_listen = 0;
    bool last = false, use_mmap = true;
    std::vector<float> tsplit;
    std::vector<std::pair<std::string,std::string>> ot_specs;   // (regex, buft-name) from --override-tensor/--cpu-moe
    int split_mode = -1;   // -1 = model default (layer); set via --split-mode
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](const char * nm) -> const char * { if (i+1>=argc){fprintf(stderr,"stage: %s needs arg\n",nm);exit(1);} return argv[++i]; };
        if      (a=="-m")           model_path = nx("-m");
        else if (a=="--prompt")     prompt     = nx("--prompt");
        else if (a=="--in")         in_path    = nx("--in");
        else if (a=="--out")        out_path   = nx("--out");
        else if (a=="--role")       role       = nx("--role");
        else if (a=="--connect")    connect_to = nx("--connect");
        else if (a=="--listen")     listen_port= atoi(nx("--listen"));
        else if (a=="--slots")      slots      = atoi(nx("--slots"));
        else if (a=="-ngl")         ngl        = atoi(nx("-ngl"));
        else if (a=="--n-ctx")      n_ctx      = atoi(nx("--n-ctx"));
        else if (a=="--n-ubatch")   n_ubatch   = atoi(nx("--n-ubatch"));
        else if (a=="--amb")        amb        = atoi(nx("--amb"));
        else if (a=="--return-listen") return_listen = atoi(nx("--return-listen"));
        else if (a=="--max-tokens") max_tokens = atoi(nx("--max-tokens"));
        else if (a=="--last")       last       = true;
        else if (a=="--no-mmap")    use_mmap   = false;
        else if (a=="--split-mode") { std::string m = nx("--split-mode"); split_mode = (m=="tensor")?3:(m=="row")?2:(m=="none")?0:1; }
        else if (a=="--tensor-split") { std::string ts = nx("--tensor-split"); size_t p=0,c;
            while ((c=ts.find(',',p))!=std::string::npos) { tsplit.push_back(atof(ts.substr(p,c-p).c_str())); p=c+1; }
            tsplit.push_back(atof(ts.substr(p).c_str())); }
        else if (a=="-ot" || a=="--override-tensor") {   // "regex=buft,regex2=buft2" (only =CPU supported); frees VRAM
            std::string v = nx("-ot"); size_t p = 0;
            while (p <= v.size()) {
                size_t comma = v.find(',', p);
                std::string one = v.substr(p, comma==std::string::npos ? std::string::npos : comma-p);
                size_t eq = one.find('=');
                if (eq != std::string::npos && eq > 0) ot_specs.push_back({ one.substr(0,eq), one.substr(eq+1) });
                else if (!one.empty()) fprintf(stderr, "stage: ignoring malformed -ot spec '%s'\n", one.c_str());
                if (comma==std::string::npos) break; p = comma+1;
            }
        }
        else if (a=="-cmoe" || a=="--cpu-moe") {   // offload ALL expert (MoE) tensors in this stage's window to CPU
            ot_specs.push_back({ "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", "CPU" });
        }
        else { fprintf(stderr,"stage: unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) { fprintf(stderr,"stage: -m required\n"); return 1; }
    fprintf(stderr, "stage: IL=[%s,%s) EMIT=%s role=%s slots=%d\n",
            getenv("STAGE_IL_START")?getenv("STAGE_IL_START"):"-",
            getenv("STAGE_IL_END")?getenv("STAGE_IL_END"):"-",
            getenv("STAGE_EMIT")?getenv("STAGE_EMIT"):"-", role.empty()?"file":role.c_str(), slots);

    // ik_llama uses a static backend registry (built-in backends); there is NO
    // ggml_backend_load_all() to dynamically load backend shared libs. llama_backend_init()
    // initializes the registered backends. (The mainline ggml_backend_load_all() call here
    // was the dynamic-backend equivalent and does not exist in ik_llama.)
    llama_backend_init();

    // Resolve --override-tensor / --cpu-moe specs into a NULL-terminated buft-override list.
    // Pattern strings are owned by ot_pat_storage (reserved up-front so c_str() pointers stay
    // stable). CPU-only by design (the use case is freeing VRAM).
    std::vector<std::string> ot_pat_storage; ot_pat_storage.reserve(ot_specs.size());
    std::vector<llama_model_tensor_buft_override> buft_ovr;
    if (!ot_specs.empty()) {
        for (auto & sp : ot_specs) ot_pat_storage.push_back(sp.first);   // finalize storage first (no realloc after)
        for (size_t i = 0; i < ot_specs.size(); ++i) {
            const std::string & bn = ot_specs[i].second;
            if (bn != "CPU" && bn != "cpu") { fprintf(stderr, "stage: -ot only supports '=CPU' (got '%s')\n", bn.c_str()); return 1; }
            buft_ovr.push_back({ ot_pat_storage[i].c_str(), ggml_backend_cpu_buffer_type() });
            fprintf(stderr, "stage: buft-override '%s' -> CPU\n", ot_pat_storage[i].c_str());
        }
        buft_ovr.push_back({ nullptr, nullptr });   // NULL terminator
    }
    model_bundle b;
    g_amb = amb;
    if (!load(b, model_path, ngl, n_ctx, use_mmap, tsplit, split_mode, n_ubatch, buft_ovr)) return 1;

    if (role == "head" && getenv("STAGE_MTP") && return_listen > 0) {
        // ===== MTP verify head (sequential; mirrors mainline stage-server gen()) =====
        // Downstream mainline tail runs run_tail_pipe_mtp (STAGE_MTP=1): it samples, drafts a
        // k-chain via NextN, and returns mtp_msg records on the DIRECT return socket. We issue
        // the dictated verify wave [conf, d_1..d_k] at p_base.., trimming our own KV for the
        // re-issued positions (rejected-draft rollback) before each decode.
        int plen = -llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> ptoks(plen);
        llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, true);
        size_t colon = connect_to.find(':');
        int fd = tcp_connect(connect_to.substr(0,colon), atoi(connect_to.substr(colon+1).c_str()));
        if (fd < 0) return 1;
        int rfd = tcp_listen_accept(return_listen);
        if (rfd < 0) { fprintf(stderr, "stage[head/mtp]: return accept failed\n"); return 1; }
        std::vector<int32_t> tok, seq, pos;
        for (int p = 0; p < plen; ++p) { tok.push_back(ptoks[p]); seq.push_back(0); pos.push_back(p); }
        hidden_blob h;
        auto pt0 = std::chrono::steady_clock::now();
        if (!run_tokens(b, tok, seq, pos, h)) { fprintf(stderr,"stage[head/mtp]: prefill failed\n"); return 1; }
        if (!send_hidden(fd, h)) return 1;
        mtp_msg m;
        if (!recv_mtp_msg(rfd, m)) { fprintf(stderr,"stage[head/mtp]: prefill return failed\n"); return 1; }
        double psec = std::chrono::duration<double>(std::chrono::steady_clock::now() - pt0).count();
        fprintf(stderr, "stage[head/mtp]: PREFILL %d tok in %.2fs = %.1f tok/s (full-ring)\n", plen, psec, plen/psec);
        long n_out = 0, n_waves = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            n_out += m.n_new;
            if (getenv("STAGE_PRINT")) for (int t : m.out) print_piece(b, t, 0);
            if (m.eog || m.issue.empty() || n_out >= max_tokens) break;
            std::vector<int32_t> dt = m.issue, ds(m.issue.size(), 0), dp(m.issue.size());
            for (size_t i = 0; i < m.issue.size(); ++i) dp[i] = m.p_base + (int) i;
            llama_kv_cache_seq_rm(b.ctx, 0, m.p_base, -1);   // rollback rejected-draft KV in OUR window
            hidden_blob hh;
            if (!run_tokens(b, dt, ds, dp, hh)) { fprintf(stderr,"stage[head/mtp]: verify decode failed\n"); break; }
            if (!send_hidden(fd, hh)) break;
            if (!recv_mtp_msg(rfd, m)) break;
            n_waves++;
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "stage[head/mtp]: %ld tokens in %ld waves in %.2fs = %.2f tok/s (%.2f tok/wave)\n",
                n_out, n_waves, sec, sec > 0 ? n_out/sec : 0.0, n_waves > 0 ? (double) n_out/n_waves : 0.0);
    } else if (role == "head") {
        // tokenize the prompt once; replicate across `slots` sequences
        int plen = -llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> ptoks(plen);
        llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, true);
        size_t colon = connect_to.find(':');
        int fd = tcp_connect(connect_to.substr(0,colon), atoi(connect_to.substr(colon+1).c_str()));
        if (fd < 0) return 1;
        // prefill: every slot gets the full prompt (seq-contiguous layout)
        std::vector<int32_t> tok, seq, pos;
        for (int s = 0; s < slots; ++s) for (int p = 0; p < plen; ++p) { tok.push_back(ptoks[p]); seq.push_back(s); pos.push_back(p); }
        hidden_blob h;
        if (!run_tokens(b, tok, seq, pos, h)) { fprintf(stderr,"stage[head]: prefill failed\n"); return 1; }
        if (!send_hidden(fd, h)) return 1;
        std::vector<int32_t> n_past(slots, plen);
        fprintf(stderr, "stage[head]: prefilled %d slots x %d tok\n", slots, plen);
        int hsteps = 0;
        auto ht0 = std::chrono::steady_clock::now();   // time decode loop (reliable readback)
        for (;;) {
            std::vector<int32_t> back; int32_t eog;
            if (!recv_tokens(fd, back, eog) || eog) break;
            std::vector<int32_t> dt, ds, dp;
            for (int s = 0; s < slots; ++s) { dt.push_back(back[s]); ds.push_back(s); dp.push_back(n_past[s]++); }
            hidden_blob hd;
            if (!run_tokens(b, dt, ds, dp, hd)) break;
            if (!send_hidden(fd, hd)) break;
            hsteps++;
        }
        double hsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ht0).count();
        double hagg = (double) hsteps * slots / hsec;
        fprintf(stderr, "stage[head]: DECODE %d steps x %d slots in %.2fs = %.2f tok/s agg, %.2f t/s/slot\n",
                hsteps, slots, hsec, hagg, hagg / slots);
    } else if (role == "tail") {
      for (;;) {                                       // loop-accept: serve c1/c4/c8 without reload
        int fd = tcp_listen_accept(listen_port); if (fd < 0) break;
        llama_kv_cache_clear(b.ctx); // fresh KV per client
        hidden_blob h, dummy;
        if (!recv_hidden(fd, h)) { close(fd); continue; }
        if (!run_hidden(b, h, false, dummy)) { fprintf(stderr,"stage[tail]: prefill failed\n"); close(fd); continue; }
        // output rows == input rows; the last row of each seq carries its logits.
        int C = 0; for (int v : h.seq) C = (v+1 > C) ? v+1 : C;
        std::vector<int> lastrow(C, -1);
        for (int i = 0; i < h.n_rows; ++i) lastrow[h.seq[i]] = i;
        std::vector<int32_t> tok(C);
        for (int s = 0; s < C; ++s) { tok[s] = argmax_ith(b, lastrow[s]); }
        if (getenv("STAGE_PRINT")) for (int s = 0; s < C; ++s) print_piece(b, tok[s], s);
        int gen = 1, dsteps = 0;
        auto t0 = std::chrono::steady_clock::now();   // time the decode loop only (prefill excluded)
        while (gen < max_tokens) {
            bool all_eog = true; for (int s=0;s<C;++s) if (!llama_vocab_is_eog(b.vocab, tok[s])) all_eog=false;
            if (all_eog && !getenv("STAGE_IGNORE_EOG")) break;
            if (!send_tokens(fd, tok, 0)) break;
            hidden_blob hd;
            if (!recv_hidden(fd, hd)) break;
            if (!run_hidden(b, hd, false, dummy)) break;
            for (int s = 0; s < C; ++s) tok[s] = argmax_ith(b, s);
            if (getenv("STAGE_PRINT")) for (int s = 0; s < C; ++s) print_piece(b, tok[s], s);
            gen++; dsteps++;
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        send_tokens(fd, tok, 1);
        double agg = (double) dsteps * C / sec;
        fprintf(stderr, "stage[tail]: decode %d steps x %d slots in %.2fs = %.2f tok/s agg, %.2f t/s/slot\n",
                dsteps, C, sec, agg, agg / C);
        close(fd);
      }
    } else if (role == "relay") {
        // chain middle stage: recv hidden from upstream -> run window -> forward to
        // downstream; relay sampled tokens back upstream. Lets N stages chain
        // head -> relay -> ... -> tail (each relay --listen <up> --connect <down>).
        size_t c = connect_to.find(':');
        std::string dh = connect_to.substr(0, c); int dp = atoi(connect_to.substr(c + 1).c_str());
        for (;;) {
            int fd_up = tcp_listen_accept(listen_port); if (fd_up < 0) break;
            int fd_down = tcp_connect(dh, dp); if (fd_down < 0) { close(fd_up); continue; }
            llama_kv_cache_clear(b.ctx);
            hidden_blob in, out; long steps = 0;
            for (;;) {
                if (!recv_hidden(fd_up, in)) break;
                if (!run_hidden(b, in, /*emit=*/true, out)) break;
                if (!send_hidden(fd_down, out)) break;
                std::vector<int32_t> toks; int32_t eog;
                if (!recv_tokens(fd_down, toks, eog)) break;
                if (!send_tokens(fd_up, toks, eog)) break;
                steps++;
                if (eog) break;
            }
            close(fd_up); close(fd_down);
            fprintf(stderr, "stage[relay]: client done (%ld steps)\n", steps);
        }
    } else {
        // FILE mode (single sequence)
        const bool emit = !out_path.empty();
        hidden_blob hin, hout; bool ok;
        if (in_path.empty()) {
            int plen = -llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
            std::vector<llama_token> ptoks(plen);
            llama_vocab_tokenize(b.vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, true);
            std::vector<int32_t> tok(ptoks.begin(), ptoks.end()), seq(plen, 0), pos(plen);
            for (int i = 0; i < plen; ++i) pos[i] = i;
            ok = run_tokens(b, tok, seq, pos, hout);
        } else {
            if (!read_file(in_path.c_str(), hin)) return 1;
            if (hin.n_embd != b.n_embd) { fprintf(stderr,"stage: n_embd mismatch\n"); return 1; }
            ok = run_hidden(b, hin, emit, hout);
        }
        if (!ok) { fprintf(stderr,"stage: decode failed\n"); return 1; }
        if (emit) { if (!write_file(out_path.c_str(), hout)) return 1;
                    fprintf(stderr,"stage: wrote %d x %d hidden -> %s\n", hout.n_rows, hout.n_embd, out_path.c_str()); }
        if (last) {
            const float * lg = llama_get_logits_ith(b.ctx, -1);
            int best = 0; for (int v=1; v<b.n_vocab; ++v) if (lg[v]>lg[best]) best=v;
            char buf[256]; int np = llama_token_to_piece_vocab(b.vocab, best, buf, sizeof(buf), 0, true);
            printf("ARGMAX token=%d logit=%.4f piece=%s\n", best, lg[best], std::string(buf, np>0?np:0).c_str());
        }
    }
    llama_free(b.ctx); llama_free_model(b.model);
    return 0;
}
