#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rma-channel.h"

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

// Cross-platform socket with optional RDMA tiers (GGML_RPC_RDMA build).
//   T0: plain TCP byte stream (always available)
//   T1: RDMA SEND/RECV byte stream (auto-upgraded when the HELLO caps exchange
//       finds a RoCE device matching the socket's local address on both ends)
//   T2: "Scope-B": control plane stays on TCP, the forward hot path becomes
//       one-sided RDMA WRITE_WITH_IMM into pre-registered slot tables
//       (see rma-channel.h). Negotiated via a capability bit in the caps.
// All tiers are byte-stream compatible: peers that never negotiate stay on T0.
struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // --- Scope-B (T2) data plane -------------------------------------------
    // True once the RC QP has been brought up (RTS) by update_caps, i.e. T1 was
    // reachable and the QP is usable for one-sided WRITE.
    bool rdma_activated() const;
    // True if the peer advertised the Scope-B capability bit in its conn_caps.
    bool peer_scope_b() const;
    // True if BOTH peers want Scope-B (symmetric: peer bit AND local opt-in), so
    // the control plane is on TCP and the slot-table handshake should proceed.
    bool scope_b_intent() const;
    // Bring up the Scope-B data plane on this (already RDMA-activated) socket:
    // register the local slot table, create the rma_channel over the live QP,
    // and move the control plane back to the TCP byte stream so the QP is
    // dedicated to WRITE_WITH_IMM (clean completion queue). Returns false (and
    // leaves the socket on its prior tier) on any failure.
    bool setup_scope_b(const rma_slot_config & cfg);
    // The data-plane channel, or nullptr if Scope-B is not active on this conn.
    rma_channel * rma() const;
    // Max WRITEs the initiator may have un-drained at the peer at once (bounded
    // by the peer's pre-posted receive ring). Caller falls back to T1/T0 beyond.
    uint16_t rma_max_in_flight() const;
    // Underlying TCP control-socket fd. After setup_scope_b the control plane is
    // back on TCP and the QP is WRITE-only, so the stage-runner can ride the raw
    // fd for its token/MTP back-edge (and shutdown() it to unblock a reader).
    int fd() const;
    // True if the peer has closed/half-closed the control socket (FIN/RST/keepalive-timeout). The
    // Scope-B T2 receive busy-poll (stage_conn_read) calls this periodically so it unwedges when an
    // RDMA upstream dies (the WRITE_WITH_IMM completion never arrives over a dead QP).
    bool peer_closed() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
