#include "llama-experts-remote.h"

#include "ggml.h"
#include "llama-impl.h"

#include <cerrno>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// ---- config -----------------------------------------------------------------

// "1,3,10-19" -> marks those absolute layer ids as served by endpoint `ep`.
// Returns false on a malformed spec or on an overlap with an already-claimed layer.
static bool er_parse_layers(const char * s, int ep, std::vector<int32_t> & layer_ep) {
    const char * p = s;
    if (*p == '\0') {
        return false;
    }
    while (*p) {
        char * end = nullptr;
        long a = strtol(p, &end, 10);
        if (end == p) {
            return false;
        }
        long b = a;
        if (end && *end == '-') {
            const char * q = end + 1;
            b = strtol(q, &end, 10);
            if (end == q) {
                return false;
            }
        }
        if (a < 0 || b < a || b >= 4096) {
            return false;
        }
        if ((long) layer_ep.size() <= b) {
            layer_ep.resize(b + 1, -1);
        }
        for (long i = a; i <= b; ++i) {
            if (layer_ep[i] >= 0 && layer_ep[i] != ep) {
                LLAMA_LOG_ERROR("%s: layer %ld claimed by two expert-servers (%d and %d)\n",
                        __func__, i, (int) layer_ep[i], ep);
                return false;
            }
            layer_ep[i] = (int32_t) ep;
        }
        if (end == nullptr || *end == '\0') {
            break;
        }
        if (*end != ',') {
            return false;
        }
        p = end + 1;
    }
    return true;
}

// "host:port" -> endpoint. Returns false if there is no port.
static bool er_parse_addr(const std::string & addr, llama_experts_remote_endpoint & ep) {
    const size_t colon = addr.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= addr.size()) {
        return false;
    }
    ep.host = addr.substr(0, colon);
    ep.port = atoi(addr.c_str() + colon + 1);
    return ep.port > 0;
}

static llama_experts_remote_cfg llama_experts_remote_parse_env() {
    llama_experts_remote_cfg cfg;

    const char * env = getenv("LLAMA_EXPERTS_REMOTE");
    if (env == nullptr || env[0] == '\0') {
        return cfg;
    }

    if (const char * ke = getenv("LLAMA_EXPERTS_REMOTE_KEEP_EXPS")) {
        cfg.keep_exps = atoi(ke) != 0;
    }

    // split the endpoint list on ';' (a layer SPEC may itself contain ',')
    std::vector<std::string> parts;
    {
        const std::string s = env;
        size_t i = 0;
        while (i <= s.size()) {
            const size_t j = s.find(';', i);
            const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!tok.empty()) {
                parts.push_back(tok);
            }
            if (j == std::string::npos) {
                break;
            }
            i = j + 1;
        }
    }
    if (parts.empty()) {
        LLAMA_LOG_ERROR("%s: LLAMA_EXPERTS_REMOTE is set but empty\n", __func__);
        return cfg;
    }

    for (size_t k = 0; k < parts.size(); ++k) {
        const std::string & part = parts[k];
        const size_t at = part.find('@');

        llama_experts_remote_endpoint ep;
        if (!er_parse_addr(at == std::string::npos ? part : part.substr(0, at), ep)) {
            GGML_ABORT("experts-remote: LLAMA_EXPERTS_REMOTE entry '%s' is not host:port[@LAYERS]", part.c_str());
        }

        const int idx = (int) cfg.endpoints.size();
        cfg.endpoints.push_back(ep);

        if (at != std::string::npos) {
            if (!er_parse_layers(part.c_str() + at + 1, idx, cfg.layer_ep)) {
                GGML_ABORT("experts-remote: bad layer spec in '%s' (expect a,b,c-d; layers may not overlap)", part.c_str());
            }
        } else if (parts.size() > 1) {
            GGML_ABORT("experts-remote: entry '%s' needs an @LAYERS spec - with more than one "
                       "expert-server every endpoint must state the layers it serves", part.c_str());
        } else {
            // single endpoint, no '@' -> legacy surface
            const char * ls = getenv("LLAMA_EXPERTS_REMOTE_LAYERS");
            if (ls != nullptr && ls[0] != '\0') {
                if (!er_parse_layers(ls, idx, cfg.layer_ep)) {
                    GGML_ABORT("experts-remote: bad LLAMA_EXPERTS_REMOTE_LAYERS spec '%s'", ls);
                }
            } else {
                cfg.default_ep = (int32_t) idx;
            }
        }
    }

    cfg.enabled = true;

    for (size_t k = 0; k < cfg.endpoints.size(); ++k) {
        std::string cov;
        if ((int32_t) k == cfg.default_ep) {
            cov = "all MoE layers";
        } else {
            int lo = -1;
            char buf[64];
            for (size_t il = 0; il <= cfg.layer_ep.size(); ++il) {
                const bool mine = il < cfg.layer_ep.size() && cfg.layer_ep[il] == (int32_t) k;
                if (mine && lo < 0) {
                    lo = (int) il;
                } else if (!mine && lo >= 0) {
                    snprintf(buf, sizeof(buf), "%s%d-%d", cov.empty() ? "" : ",", lo, (int) il - 1);
                    cov += buf;
                    lo = -1;
                }
            }
            if (cov.empty()) {
                cov = "no layers";
            }
        }
        LLAMA_LOG_INFO("%s: expert-server[%d] %s:%d serves layers %s\n", __func__,
                (int) k, cfg.endpoints[k].host.c_str(), cfg.endpoints[k].port, cov.c_str());
    }

    return cfg;
}

const llama_experts_remote_cfg & llama_experts_remote_get_cfg() {
    static const llama_experts_remote_cfg cfg = llama_experts_remote_parse_env();
    return cfg;
}

// ---- transport --------------------------------------------------------------

static bool er_send_all(int fd, const void * b, size_t n) {
    const char * p = (const char *) b;
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) {
            if (k < 0 && (errno == EINTR)) continue;
            return false;
        }
        p += k; n -= (size_t) k;
    }
    return true;
}

static bool er_recv_all(int fd, void * b, size_t n) {
    char * p = (char *) b;
    while (n > 0) {
        ssize_t k = recv(fd, p, n, 0);
        if (k <= 0) {
            if (k < 0 && (errno == EINTR)) continue;
            return false;
        }
        p += k; n -= (size_t) k;
    }
    return true;
}

// Try the CAPS handshake on a freshly connected socket. Returns true on success
// (and validates the server's reported n_layer / n_embd against the caller's
// expectations), false on any transport or protocol mismatch.
static bool er_caps_handshake(int fd, const llama_experts_remote_endpoint & ep,
                              const std::vector<int32_t> & assigned_layers, int expect_n_embd) {
    // client -> server: CAPS magic + version
    int32_t client_caps[2] = { LLAMA_EXPERTS_MAGIC_CAPS, LLAMA_EXPERTS_VERSION };
    if (!er_send_all(fd, client_caps, sizeof(client_caps))) {
        LLAMA_LOG_ERROR("%s: CAPS send failed to %s:%d\n", __func__, ep.host.c_str(), ep.port);
        return false;
    }
    // server -> client: CAPS magic + version + n_served + n_embd, then the
    // served layer ids and backend/thread metadata.
    int32_t reply[4];
    if (!er_recv_all(fd, reply, sizeof(reply))) {
        LLAMA_LOG_ERROR("%s: CAPS reply failed from %s:%d\n", __func__, ep.host.c_str(), ep.port);
        return false;
    }
    if (reply[0] != LLAMA_EXPERTS_MAGIC_CAPS) {
        LLAMA_LOG_ERROR("%s: bad CAPS magic from %s:%d: %08x\n", __func__, ep.host.c_str(), ep.port, reply[0]);
        return false;
    }
    if (reply[1] != LLAMA_EXPERTS_VERSION) {
        LLAMA_LOG_ERROR("%s: version mismatch with %s:%d (client %d, server %d)\n", __func__,
                ep.host.c_str(), ep.port, LLAMA_EXPERTS_VERSION, reply[1]);
        return false;
    }
    if (reply[2] < 0 || reply[2] > 4096) {
        LLAMA_LOG_ERROR("%s: invalid served-layer count from %s:%d: %d\n", __func__, ep.host.c_str(), ep.port, reply[2]);
        return false;
    }
    if (reply[3] != expect_n_embd) {
        LLAMA_LOG_ERROR("%s: embedding mismatch with %s:%d (server %d, client %d)\n", __func__,
                ep.host.c_str(), ep.port, reply[3], expect_n_embd);
        return false;
    }
    std::vector<int32_t> caps((size_t) reply[2] + 2);
    if (!er_recv_all(fd, caps.data(), caps.size() * sizeof(int32_t))) return false;
    std::vector<int32_t> served(caps.begin(), caps.begin() + reply[2]);
    for (int32_t layer : assigned_layers) {
        if (std::find(served.begin(), served.end(), layer) == served.end()) {
            LLAMA_LOG_ERROR("%s: layer assignment mismatch with %s:%d (assigned layer %d is absent from server list)\n",
                    __func__, ep.host.c_str(), ep.port, layer);
            return false;
        }
    }
    return true;
}

static int er_retry_ms() {
    const char * env = getenv("LLAMA_EXPERTS_REMOTE_RETRY_MS");
    if (env == nullptr || *env == '\0') return 500;
    const long value = strtol(env, nullptr, 10);
    return value > 0 && value <= 8000 ? (int) value : 500;
}

static bool er_wait_retry(int attempt, const llama_experts_remote_endpoint & ep) {
    const int base = er_retry_ms();
    const int delay = std::min(8000, base << std::min(attempt - 1, 3));
    LLAMA_LOG_WARN("experts-remote: retry %d to %s:%d in %d ms\n", attempt, ep.host.c_str(), ep.port, delay);
    usleep((useconds_t) delay * 1000);
    return true;
}

static int llama_experts_remote_connect(const llama_experts_remote_endpoint & ep) {
    struct addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", ep.port);

    struct addrinfo * res = nullptr;
    int rc = getaddrinfo(ep.host.c_str(), portstr, &hints, &res);
    if (rc != 0 || res == nullptr) {
        LLAMA_LOG_ERROR("%s: cannot resolve expert-server %s:%d: %s\n", __func__,
                ep.host.c_str(), ep.port, gai_strerror(rc));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo * ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd >= 0) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        LLAMA_LOG_INFO("%s: connected to expert-server %s:%d\n", __func__, ep.host.c_str(), ep.port);
    } else {
        LLAMA_LOG_ERROR("%s: cannot connect to expert-server %s:%d: %s\n", __func__,
                ep.host.c_str(), ep.port, strerror(errno));
    }
    return fd;
}

// one connection per endpoint per process; calls are serialized (the residual
// dependency does that anyway within a graph; the mutex covers multi-context use)
static std::mutex er_mutex;

// NOTE: the exit reporter must NOT touch the parsed config. cfg is a
// function-local static constructed during model load, i.e. AFTER this
// translation unit's namespace-scope statics, so it is destroyed BEFORE the
// atexit handler runs - reading it there is a use-after-destruction (it printed
// an empty host and segfaulted on the CPU-attention path). Everything the
// report needs is therefore copied into er_conn at connect time, and the label
// is a plain char buffer so it owns no heap.
// Phase breakdown of one EXPERT_CALL, kept per endpoint and split by shape.
// Decode calls carry one row and a 24 KiB frame; prefill calls carry n_ubatch
// rows and a much larger frame, and the two behave nothing alike -
// averaging them together is what hid the prefill stall in the first place.
struct er_phase {
    uint64_t n        = 0;
    uint64_t rows     = 0;
    uint64_t t_asm    = 0;   // build the frame (includes the per-call allocation)
    uint64_t t_send   = 0;   // send_all of the request
    uint64_t t_wait   = 0;   // blocked until the 16-byte reply header arrives
    uint64_t t_recv   = 0;   // pull the reply payload
    uint64_t t_total  = 0;
    uint64_t t_min    = ~0ull;
};

struct er_conn {
    int      fd        = -1;
    char     label[80] = {0};   // "host:port", filled on connect
    er_phase dec;               // n_tok == 1
    er_phase pre;               // n_tok  > 1
};

static std::vector<er_conn> er_conns;

// per-endpoint stats (send-start to payload-received, i.e. wire + server time);
// summarized at process exit for the mechanism-overhead benches
static void er_report_phase(const char * label, const char * shape, const er_phase & p) {
    if (p.n == 0) {
        return;
    }
    const double n = (double) p.n;
    fprintf(stderr,
            "experts-remote: %s %-7s %6llu calls %9llu rows | asm %7.1f  send %8.1f  wait %9.1f  recv %8.1f  = %9.1f us/call"
            " | min %8.1f | total %8.1f s\n",
            label, shape,
            (unsigned long long) p.n, (unsigned long long) p.rows,
            p.t_asm / n, p.t_send / n, p.t_wait / n, p.t_recv / n, p.t_total / n,
            (double) p.t_min, p.t_total / 1e6);
}

static void er_report() {
    for (size_t k = 0; k < er_conns.size(); ++k) {
        const er_conn & c = er_conns[k];
        if (c.dec.n == 0 && c.pre.n == 0) {
            continue;
        }
        er_report_phase(c.label, "decode",  c.dec);
        er_report_phase(c.label, "prefill", c.pre);
    }
}

static struct er_report_at_exit_t {
    er_report_at_exit_t() { atexit(er_report); }
} er_report_at_exit;

// ---- the custom op ----------------------------------------------------------

void llama_experts_remote_custom_cb(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,   // hidden  [n_embd, n_tok] f32
        const struct ggml_tensor * b,   // ids     [n_topk, n_tok] i32 (top_k view: row stride > row size)
        const struct ggml_tensor * c,   // weights [1, n_topk, n_tok] f32
        int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const int layer = (int) (intptr_t) userdata;

    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_I32);
    GGML_ASSERT(c->type == GGML_TYPE_F32);

    const int64_t n_embd = a->ne[0];
    const int64_t n_tok  = a->ne[1];
    const int64_t n_topk = b->ne[0];

    GGML_ASSERT(b->ne[1] == n_tok);
    GGML_ASSERT(c->ne[0] == 1 && c->ne[1] == n_topk && c->ne[2] == n_tok);
    GGML_ASSERT(dst->ne[0] == n_embd && dst->ne[1] == n_tok);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const llama_experts_remote_cfg & cfg = llama_experts_remote_get_cfg();
    const int epi = cfg.endpoint_for(layer);
    GGML_ASSERT(epi >= 0 && "experts-remote custom op scheduled for an uncovered layer");
    const llama_experts_remote_endpoint & ep = cfg.endpoints[epi];

    // assemble the frame (stride-aware gathers; ids is typically a top_k view
    // whose row stride is n_expert, not n_topk)
    const size_t n_ids = (size_t) n_topk * n_tok;
    const size_t n_hid = (size_t) n_embd * n_tok;

    struct timespec ta0, ta1, ts0, ts1, tw1, tr1;
    clock_gettime(CLOCK_MONOTONIC, &ta0);

    // Reused across calls: a fresh std::vector here means a 12.58 MiB allocation
    // AND a full value-initialisation memset on every prefill call. thread_local,
    // not static - assembly happens before er_mutex is taken, so a shared buffer
    // would race if two contexts ever call concurrently.
    thread_local std::vector<uint8_t> frame;
    frame.resize(5 * sizeof(int32_t) + n_ids * (sizeof(int32_t) + sizeof(float)) + n_hid * sizeof(float));
    uint8_t * w = frame.data();

    const int32_t hdr[5] = { LLAMA_EXPERTS_MAGIC_CALL, (int32_t) layer, (int32_t) n_tok, (int32_t) n_topk, (int32_t) n_embd };
    memcpy(w, hdr, sizeof(hdr)); w += sizeof(hdr);

    for (int64_t t = 0; t < n_tok; ++t) {
        const char * src = (const char *) b->data + t * b->nb[1];
        for (int64_t k = 0; k < n_topk; ++k) {
            memcpy(w, src + k * b->nb[0], sizeof(int32_t));
            w += sizeof(int32_t);
        }
    }
    for (int64_t t = 0; t < n_tok; ++t) {
        const char * src = (const char *) c->data + t * c->nb[2];
        for (int64_t k = 0; k < n_topk; ++k) {
            memcpy(w, src + k * c->nb[1], sizeof(float));
            w += sizeof(float);
        }
    }
    for (int64_t t = 0; t < n_tok; ++t) {
        const char * src = (const char *) a->data + t * a->nb[1];
        memcpy(w, src, (size_t) n_embd * sizeof(float));
        w += (size_t) n_embd * sizeof(float);
    }
    GGML_ASSERT(w == frame.data() + frame.size());

    clock_gettime(CLOCK_MONOTONIC, &ta1);

    std::lock_guard<std::mutex> lock(er_mutex);

    if (er_conns.size() < cfg.endpoints.size()) {
        er_conns.resize(cfg.endpoints.size());
    }
    er_conn & conn = er_conns[epi];

    // Establish (or re-establish) the connection, including the CAPS handshake.
    // The server reports its n_layer and n_embd; we validate them against the
    // model's actual values so a mismatched server is caught at first call,
    // not silently producing garbage.
    const int expect_n_embd  = (int) n_embd;
    std::vector<int32_t> assigned_layers;
    if (cfg.default_ep != epi) {
        for (size_t i = 0; i < cfg.layer_ep.size(); ++i) if (cfg.layer_ep[i] == epi) assigned_layers.push_back((int32_t) i);
    }

    auto er_establish = [&]() -> bool {
        if (conn.fd >= 0) {
            close(conn.fd);
            conn.fd = -1;
        }
        conn.fd = llama_experts_remote_connect(ep);
        if (conn.fd < 0) {
            return false;
        }
        snprintf(conn.label, sizeof(conn.label), "%s:%d", ep.host.c_str(), ep.port);
        if (!er_caps_handshake(conn.fd, ep, assigned_layers, expect_n_embd)) {
            close(conn.fd); conn.fd = -1;
            return false;
        }
        return true;
    };

    // On any wire/protocol failure, tear down the connection and retry the
    // whole RPC once after re-establishing. This turns transient socket errors
    // (a momentary EPIPE, a server that was briefly unreachable) into a retried
    // call instead of a process abort - the residual dependency means a single
    // retry is enough to ride through a server restart or a dropped connection.
    int retries = 0;
    bool ok = false;
    for (;;) {
        if (conn.fd < 0) {
            if (!er_establish()) {
                if (retries++ < 3) {
                    er_wait_retry(retries, ep);
                    continue;
                }
                GGML_ABORT("experts-remote: cannot reach expert-server %s:%d after %d retries",
                           ep.host.c_str(), ep.port, retries);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &ts0);

        int32_t rhdr[4];
        if (!er_send_all(conn.fd, frame.data(), frame.size())) {
            LLAMA_LOG_WARN("experts-remote: send failed to %s:%d (layer %d), retrying\n",
                           ep.host.c_str(), ep.port, layer);
            close(conn.fd); conn.fd = -1;
            if (retries++ < 3) { er_wait_retry(retries, ep); continue; }
            GGML_ABORT("experts-remote: wire failure talking to %s:%d (layer %d)",
                       ep.host.c_str(), ep.port, layer);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (!er_recv_all(conn.fd, rhdr, sizeof(rhdr))) {
            LLAMA_LOG_WARN("experts-remote: recv header failed from %s:%d (layer %d), retrying\n",
                           ep.host.c_str(), ep.port, layer);
            close(conn.fd); conn.fd = -1;
            if (retries++ < 3) { er_wait_retry(retries, ep); continue; }
            GGML_ABORT("experts-remote: wire failure talking to %s:%d (layer %d)",
                       ep.host.c_str(), ep.port, layer);
        }
        clock_gettime(CLOCK_MONOTONIC, &tw1);
        if (rhdr[0] != LLAMA_EXPERTS_MAGIC_RET || rhdr[1] != layer || rhdr[2] != n_tok) {
            LLAMA_LOG_WARN("experts-remote: bad EXPERT_RET header from %s:%d (layer %d), retrying\n",
                           ep.host.c_str(), ep.port, layer);
            close(conn.fd); conn.fd = -1;
            if (retries++ < 3) { er_wait_retry(retries, ep); continue; }
            GGML_ABORT("experts-remote: bad EXPERT_RET header from %s:%d (layer %d)",
                       ep.host.c_str(), ep.port, layer);
        }
        if (rhdr[3] != 0) {
            LLAMA_LOG_WARN("experts-remote: expert-server reported failure %d (layer %d), retrying\n",
                           rhdr[3], layer);
            close(conn.fd); conn.fd = -1;
            if (retries++ < 3) { er_wait_retry(retries, ep); continue; }
            GGML_ABORT("experts-remote: expert-server reported failure %d (layer %d)", rhdr[3], layer);
        }
        if (!er_recv_all(conn.fd, dst->data, n_hid * sizeof(float))) {
            LLAMA_LOG_WARN("experts-remote: truncated EXPERT_RET payload from %s:%d (layer %d), retrying\n",
                           ep.host.c_str(), ep.port, layer);
            close(conn.fd); conn.fd = -1;
            if (retries++ < 3) continue;
            GGML_ABORT("experts-remote: truncated EXPERT_RET payload from %s:%d (layer %d)",
                       ep.host.c_str(), ep.port, layer);
        }
        ok = true;
        break;
    }
    (void) ok;

    clock_gettime(CLOCK_MONOTONIC, &tr1);

    auto us = [](const struct timespec & a, const struct timespec & b) -> uint64_t {
        return (uint64_t) ((b.tv_sec - a.tv_sec) * 1000000LL + (b.tv_nsec - a.tv_nsec) / 1000LL);
    };
    er_phase & ph = (n_tok == 1) ? conn.dec : conn.pre;
    const uint64_t dt = us(ts0, tr1);
    ph.n      += 1;
    ph.rows   += (uint64_t) n_tok;
    ph.t_asm  += us(ta0, ta1);
    ph.t_send += us(ts0, ts1);
    ph.t_wait += us(ts1, tw1);
    ph.t_recv += us(tw1, tr1);
    ph.t_total += dt;
    if (dt < ph.t_min) {
        ph.t_min = dt;
    }
}
