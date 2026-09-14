// RoCEv2 RDMA (Scope-B) transport for the stage-runner forward path.
// Self-contained: includes transport.h (its socket_t) but NOT httplib, so the
// stage-runner's httplib `socket_t=int` never collides. Logic mirrors the
// validated isolation test (loopback + cross-vendor, 74KB WRITE byte-correct).
#include "stage_rdma.h"
#include "ggml-rpc/transport.h"
#include "ggml-rpc/rma-channel.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

struct stage_conn {
    socket_ptr   sock;
    rma_channel* rma = nullptr;   // non-null => forward path is RDMA
    rma_slot_config cfg;
    uint16_t     bulk_rr = 0;     // round-robin over the bulk slots (prefill)
    uint16_t     dec_rr  = 0;     // round-robin over the decode slots (steady-state waves)
};

static bool g_rdma = []{ const char* e = getenv("STAGE_RDMA"); return e && atoi(e) > 0; }();

// SLOTS_ANNOUNCE over the TCP byte stream: | u16 n | rma_region[n] |.
static bool exch_slots(socket_t* s, rma_channel* ch) {
    uint16_t n = ch->n_slots();
    std::vector<rma_region> loc(n);
    for (uint16_t i = 0; i < n; ++i) loc[i] = ch->local_slot(i);
    if (!s->send_data(&n, sizeof n) || !s->send_data(loc.data(), n * sizeof(rma_region))) return false;
    uint16_t rn = 0;
    if (!s->recv_data(&rn, sizeof rn)) return false;
    std::vector<rma_region> rem(rn);
    if (rn && !s->recv_data(rem.data(), rn * sizeof(rma_region))) return false;
    ch->set_remote_slots(rem.data(), rn);
    return true;
}
// caps HELLO over the byte stream (symmetric); connector sends first, listener recvs first.
static bool caps_hello(socket_t* s, bool client_first) {
    uint8_t local[RPC_CONN_CAPS_SIZE], remote[RPC_CONN_CAPS_SIZE];
    s->get_caps(local);
    if (client_first) { if (!s->send_data(local, sizeof local) || !s->recv_data(remote, sizeof remote)) return false; }
    else              { if (!s->recv_data(remote, sizeof remote) || !s->send_data(local, sizeof local)) return false; }
    s->update_caps(remote);   // brings the RC QP to RTS if both peers are RDMA-capable
    return true;
}
// Bring up Scope-B on a connected socket_t (caps already exchanged). On any
// failure leaves c->rma null so the caller stays on the TCP byte stream.
static void activate(stage_conn* c) {
    if (!g_rdma || !c->sock || !c->sock->scope_b_intent()) return;
    if (!c->sock->setup_scope_b(c->cfg)) { fprintf(stderr, "stage_rdma: setup_scope_b failed -> TCP\n"); return; }
    rma_channel* ch = c->sock->rma();
    if (!ch || !exch_slots(c->sock.get(), ch)) { fprintf(stderr, "stage_rdma: slot exchange failed -> TCP\n"); return; }
    ch->master_slots.init(c->cfg.n_decode);
    c->rma = ch;
    fprintf(stderr, "stage_rdma: RDMA ACTIVE (forward = RoCEv2 one-sided WRITE)\n");
}

stage_conn* stage_conn_connect(const char* host, int port) {
    stage_conn* c = new stage_conn();
    c->sock = socket_t::connect(host, port);
    if (c->sock && caps_hello(c->sock.get(), /*client_first=*/true)) activate(c);
    else if (c->sock) { /* TCP only */ }
    if (!c->sock) { delete c; return nullptr; }
    return c;
}
stage_conn* stage_conn_listen(int port) {
    stage_conn* c = new stage_conn();
    socket_ptr srv = socket_t::create_server("0.0.0.0", port);
    if (!srv) { delete c; return nullptr; }
    fprintf(stderr, "stage: listening on :%d\n", port);
    c->sock = srv->accept();
    if (c->sock && caps_hello(c->sock.get(), /*client_first=*/false)) activate(c);
    if (!c->sock) { delete c; return nullptr; }
    return c;
}
bool stage_conn_ok(stage_conn* c)        { return c && c->sock; }
bool stage_conn_is_rdma(stage_conn* c)   { return c && c->rma != nullptr; }
int  stage_conn_fd(stage_conn* c)        { return (c && c->sock) ? c->sock->fd() : -1; }
void stage_conn_close(stage_conn* c)     { if (c) { c->sock.reset(); delete c; } }
bool stage_conn_send(stage_conn* c, const void* b, size_t n) { return c && c->sock && c->sock->send_data(b, n); }
bool stage_conn_recv(stage_conn* c, void* b, size_t n)       { return c && c->sock && c->sock->recv_data(b, n); }
uint32_t stage_conn_slot_cap(stage_conn* c) { return (c && c->rma) ? c->cfg.bulk_slot_size : 0; }   // largest single WRITE (bulk)

bool stage_conn_write(stage_conn* c, const void* buf, uint32_t len) {
    if (!c || !c->rma) return false;
    int slot;
    if (len <= c->cfg.decode_slot_size) {                         // steady-state decode wave -> round-robin the decode ring
        // Round-robin over all n_decode (64) slots. The ring depth (64) is far larger
        // than the max waves a forward conn ever has unconsumed (<= pipeline depth,
        // typ. <= 12), so a slot is never reused before the receiver has polled it
        // (poll_writes returns slot_ptr(imm_slot), so each wave MUST keep a distinct
        // slot until consumed). The previous acquire()/in_flight_ scheme had no
        // matching release(), so in_flight_ leaked +1 per wave and the conn wedged
        // after rma_max_in_flight (24) WRITEs -> the depth>1 deadlock (more fillers =>
        // more WRITEs/token => wedged sooner: d6@5, d3@10, d1@38). Immediate release
        // is ALSO wrong: LIFO reuse hands every wave the same slot, so depth>1 waves
        // collide in one slot before the receiver reads them (wedged d6 @ 1 tok).
        slot = c->dec_rr++ % c->cfg.n_decode;
    } else if (len <= c->cfg.bulk_slot_size) {                    // prefill -> round-robin a bulk slot
        slot = c->cfg.n_decode + (c->bulk_rr++ % c->cfg.n_bulk);
    } else {
        return false;                                            // larger than a bulk slot -> caller falls back to TCP
    }
    uint8_t* p = c->rma->slot_ptr((uint16_t)slot);
    memcpy(p, buf, len);
    uint32_t imm = rma_pack_imm((uint16_t)slot, c->rma->next_seq++);
    return c->rma->write_slot((uint16_t)slot, p, len, imm);
}
const void* stage_conn_read(stage_conn* c, uint32_t* len) {
    if (!c || !c->rma) return nullptr;
    rma_completion cc;
    for (uint64_t s = 0; ; ++s) {
        int n = c->rma->poll_writes(&cc, 1);
        if (n < 0)  return nullptr;
        if (n == 0) {
            // No WRITE_WITH_IMM yet. Periodically check the control socket: if the peer died, the
            // completion will NEVER arrive over the dead QP, so without this the T2 receive busy-spins
            // forever and the relay wedges (never re-accepts the next head). THE relay recv-wedge fix.
            if ((s & 0x3FFFF) == 0 && s > 0 && c->sock && c->sock->peer_closed()) return nullptr;
            continue;                               // spin for the wave
        }
        if (len) *len = cc.byte_len;
        return c->rma->slot_ptr(rma_imm_slot(cc.imm));
    }
}
