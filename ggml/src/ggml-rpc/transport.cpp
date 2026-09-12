#include "transport.h"
#include "ggml-impl.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <array>
#include <mutex>
#include <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

// Logging shims matching this fork's ggml-rpc.cpp conventions (its ggml-impl.h
// predates the GGML_LOG_* helpers). Guarded so an older/newer ggml that does
// define them still wins.
#ifndef GGML_LOG_ERROR
#  define GGML_LOG_ERROR(...) printf(__VA_ARGS__)
#endif
#ifndef GGML_LOG_INFO
#  define GGML_LOG_INFO(...) printf(__VA_ARGS__)
#endif
#ifndef GGML_LOG_DEBUG
#  define GGML_LOG_DEBUG(...) printf(__VA_ARGS__)
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr int    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;

    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

// Local opt-in to Scope-B (T2), read once from the environment. The cap bit is
// only advertised when set; negotiation also requires the peer's bit (SPEC §C.6).
static bool rpc_local_wants_scope_b() {
    static const char * p = std::getenv("GGML_RPC_PROTOCOL");
    static bool want = (p != nullptr) && (strstr(p, "v2") != nullptr);
    return want;
}

// -----------------------------------------------------------------------------
// ibverbs implementation of the Scope-B data-plane trait (SPEC §B.2).
// Reuses the connection's already-activated RC QP (rdma_conn): the QP was
// created with IBV_ACCESS_REMOTE_WRITE and the receive ring is already posted by
// rdma_activate(), so one-sided WRITE_WITH_IMM works with no extra QP bring-up.
// The local slot table is a single contiguous, host-pinned, REMOTE_WRITE MR.
// -----------------------------------------------------------------------------
struct rma_channel_ibverbs : rma_channel {
    rdma_conn *             conn;      // borrowed (owned by socket_t::impl::rdma)
    rma_slot_config         cfg;
    void *                  region    = nullptr; // local pinned slot region
    struct ibv_mr *         region_mr = nullptr;
    std::vector<rma_region> remote;               // peer slots = WRITE targets

    explicit rma_channel_ibverbs(rdma_conn * c) : conn(c) {}
    ~rma_channel_ibverbs() override { shutdown(); }

    bool register_slots(const rma_slot_config & c) override {
        cfg = c;
        const size_t total = cfg.total_size();
        region = aligned_alloc(4096, total);
        if (!region) {
            return false;
        }
        memset(region, 0, total);
        region_mr = ibv_reg_mr(conn->pd, region, total,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!region_mr) {
            free(region);
            region = nullptr;
            return false;
        }
        GGML_LOG_INFO("RMA slots registered: %u decode×%uKiB + %u bulk×%uKiB = %.1f MiB, rkey=0x%x\n",
                      cfg.n_decode, cfg.decode_slot_size / 1024, cfg.n_bulk, cfg.bulk_slot_size / 1024,
                      (double) total / (1024.0 * 1024.0), region_mr->rkey);
        return true;
    }

    rma_region local_slot(uint16_t i) const override {
        rma_region r = {};
        r.addr     = (uint64_t) ((uint8_t *) region + cfg.slot_offset(i));
        r.rkey     = region_mr ? region_mr->rkey : 0;
        r.len      = cfg.slot_size(i);
        r.slot_idx = i;
        return r;
    }

    uint16_t n_slots() const override { return cfg.n_slots(); }

    void set_remote_slots(const rma_region * slots, uint16_t n) override {
        remote.assign(slots, slots + n);
    }

    uint8_t * slot_ptr(uint16_t i) override {
        return (uint8_t *) region + cfg.slot_offset(i);
    }
    uint32_t slot_capacity(uint16_t i) const override { return cfg.slot_size(i); }

    bool write_slot(uint16_t remote_slot, const void * buf, uint32_t len, uint32_t imm) override {
        if (remote_slot >= remote.size() || !region_mr) {
            return false;
        }
        const rma_region & R = remote[remote_slot];
        if (len > R.len) {
            return false;
        }
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t) buf;   // must lie within our registered region
        sge.length = len;
        sge.lkey   = region_mr->lkey;

        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.sg_list             = &sge;
        wr.num_sge             = 1;
        wr.send_flags          = IBV_SEND_SIGNALED;
        wr.imm_data            = htonl(imm);
        wr.wr.rdma.remote_addr = R.addr;
        wr.wr.rdma.rkey        = (uint32_t) R.rkey;

        if (ibv_post_send(conn->qp, &wr, &bad) != 0) {
            return false;
        }
        // Block on the local send completion only. For an RC RDMA_WRITE the send
        // completion is generated after the peer ACKs, i.e. the data is already
        // in the peer's slot memory and the peer's WRITE_WITH_IMM receive
        // completion has been generated. That is the ordering guarantee the
        // worker relies on: every WRITE for a step lands before the subsequent
        // control-plane command arrives on the (separate) TCP stream.
        for (;;) {
            struct ibv_wc wc;
            int n = ibv_poll_cq(conn->scq, 1, &wc);
            if (n < 0) {
                return false;
            }
            if (n == 0) {
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RMA write_slot wc error: status=%d (%s)\n",
                               wc.status, ibv_wc_status_str(wc.status));
                return false;
            }
            return true;
        }
    }

    int poll_writes(rma_completion * out, int max) override {
        int got = 0;
        while (got < max) {
            struct ibv_wc wc;
            int n = ibv_poll_cq(conn->rcq, 1, &wc);
            if (n < 0) {
                return -1;
            }
            if (n == 0) {
                break;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RMA poll_writes wc error: status=%d (%s)\n",
                               wc.status, ibv_wc_status_str(wc.status));
                return -1;
            }
            if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                out[got].imm      = ntohl(wc.imm_data);
                out[got].byte_len = wc.byte_len;
                got++;
            }
            // Re-post the consumed receive WR (reuses the rx-ring placeholder;
            // the WRITE payload itself landed in the slot, not this buffer).
            conn->post_rx((int) wc.wr_id);
        }
        return got;
    }

    void shutdown() override {
        if (region_mr) { ibv_dereg_mr(region_mr); region_mr = nullptr; }
        if (region)    { free(region); region = nullptr; }
        remote.clear();
    }
};

#endif // GGML_RPC_RDMA

struct socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // Scope-B (T2) accessors — always declared so the protocol layer is #ifdef-free.
    bool          rdma_activated() const;
    bool          peer_scope_b_caps() const;
    bool          scope_b_intent() const;
    bool          scope_b_setup(const rma_slot_config & cfg);
    rma_channel * rma_get() const;
    uint16_t      rma_max_in_flight() const;

#ifdef GGML_RPC_RDMA
    bool tcp_peer_closed();
    std::optional<rdma_gid_t> rdma_build_target_gid();
    bool rdma_probe();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);

    std::unique_ptr<rdma_conn>   rdma;
    rdma_local_info              rdma_local    = {};
    std::unique_ptr<rma_channel> rma_chan;              // Scope-B data plane
    bool                         peer_scope_b_ = false; // peer set the T2 cap bit
    bool                         rdma_qp_up    = false; // RC QP reached RTS
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(this->fd);
#else
    if (fd >= 0) close(this->fd);
#endif
}

#ifdef GGML_RPC_RDMA

bool socket_t::impl::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    short ev = POLLIN;
#ifdef POLLRDHUP
    ev |= POLLRDHUP;
#endif
    struct pollfd pfd = { fd, ev, 0 };
    int r = poll(&pfd, 1, 0);
    if (r <= 0) return false;
    if (pfd.revents & (POLLHUP | POLLERR)) return true;
#ifdef POLLRDHUP
    if (pfd.revents & POLLRDHUP) return true;
#endif
    // Belt-and-suspenders: if the control socket is readable, the byte could be real control data
    // OR an EOF (FIN -> CLOSE_WAIT, where POLLHUP/POLLERR are NOT set). MSG_PEEK one byte: recv()==0
    // means the peer half-closed. PEEK does not consume data, so it's safe on the live control socket
    // (real pending data peeks >0 and we return false). This makes dead-peer detection reliable even
    // when POLLRDHUP doesn't fire, which is the relay recv-wedge fix.
    if (pfd.revents & POLLIN) {
        char b;
        ssize_t n = recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return true;   // EOF: peer sent FIN
    }
    return false;
#else
    return false;
#endif
}

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

bool socket_t::impl::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    if (!target_gid) {
        return false;
    }

    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;

    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB
    uint8_t matched_port = 1;

    // Match the socket's local address to a (device, port, GID). MUST scan EVERY
    // active port of each device, not just port 1: a dual-port RoCE NIC carries a
    // different subnet on each port, so hardcoding port 1 silently misses the
    // port-2 link and falls back to TCP. GGML_RDMA_DEV (if set) still narrows to
    // one device, but a node with MULTIPLE RoCE devices needs it UNSET so the
    // right device is auto-picked per local IP.
    for (int d = 0; d < num_devs && !ibctx; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        ibv_device_attr da;
        uint8_t nports = (ibv_query_device(ctx, &da) == 0 && da.phys_port_cnt > 0) ? da.phys_port_cnt : 1;
        bool dev_matched = false;

        for (uint8_t port = 1; port <= nports && !dev_matched; port++) {
            ibv_port_attr pa;
            if (ibv_query_port(ctx, port, &pa) != 0) continue;
            if (pa.state != IBV_PORT_ACTIVE) continue;   // skip DOWN ports

            int found_gid = gid_idx;
            int found_version = IBV_GID_TYPE_IB;
            if (found_gid < 0) {
                // Find a GID on this port whose bytes equal the local TCP address
                // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1.
                int v2_idx = -1;
                int v1_idx = -1;
                for (int i = 0; i < pa.gid_tbl_len; i++) {
                    ibv_gid_entry entry = {};
                    if (ibv_query_gid_ex(ctx, port, i, &entry, 0) != 0) continue;
                    if (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) continue;
                    if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                        v2_idx = i;
                    } else if (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                        v1_idx = i;
                    }
                }
                if (v2_idx >= 0) {
                    found_gid = v2_idx;
                    found_version = IBV_GID_TYPE_ROCE_V2;
                } else if (v1_idx >= 0) {
                    found_gid = v1_idx;
                    found_version = IBV_GID_TYPE_ROCE_V1;
                }
            } else {
                // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, port, found_gid, &entry, 0) == 0) {
                    found_version = entry.gid_type;
                }
            }
            if (found_gid >= 0) {
                ibctx = ctx;
                gid_idx = found_gid;
                gid_version = found_version;
                matched_dev = dn;
                matched_port = port;
                rdma_local.path_mtu = pa.active_mtu;
                dev_matched = true;
            }
        }
        if (!dev_matched) ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;

    rdma_local.ib_port = matched_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    if (!rdma->pd) return false;

    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, nullptr, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, nullptr, 0);
    if (!rdma->scq || !rdma->rcq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->tx_buf || !rdma->rx_buf) return false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->tx_mr || !rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, rdma_local.ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    GGML_LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    return true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        struct ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // INIT -> RTR
    {
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return false;
        }
    }

    // RTR -> RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return false;
        }
    }

    GGML_LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    for (uint64_t s = 0; ; s++) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n > 0) {
            if (wc->status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            return wc->status == IBV_WC_SUCCESS;
        }
        if (n < 0) return false;
        if ((s & 0xFFFFF) == 0 && s > 0) {
            if (tcp_peer_closed()) {
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } else {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        struct ibv_wc wc;
        if (!rdma_poll(c->scq, &wc)) return false;

        src += chunk;
        rem -= chunk;
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        struct ibv_wc wc;
        if (!rdma_poll(c->rcq, &wc)) return false;

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;

        dst += got;
        rem -= got;
    }
    return true;
}

#endif // GGML_RPC_RDMA

bool socket_t::impl::send_data(const void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            GGML_LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            GGML_LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            LOG_DBG("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    rdma_local = {};
    if (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn & RPC_RMA_PSN_MASK;
        if (rpc_local_wants_scope_b()) {
            rc.psn |= RPC_RMA_CAP_SCOPE_B; // advertise T2 capability in the spare PSN bit
        }
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
        LOG_DBG("scope-b get_caps: GGML_RPC_PROTOCOL=%s want=%d psn=0x%x\n",
                std::getenv("GGML_RPC_PROTOCOL") ? std::getenv("GGML_RPC_PROTOCOL") : "(null)",
                (int) rpc_local_wants_scope_b(), rc.psn);
    } else {
        rdma.reset();
    }
#endif // GGML_RPC_RDMA
}

void socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    if (!rdma) {
        return;
    }
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, sizeof(rc));
    peer_scope_b_ = (rc.psn & RPC_RMA_CAP_SCOPE_B) != 0;
    const uint32_t remote_psn = rc.psn & RPC_RMA_PSN_MASK; // strip cap bit -> 24-bit PSN
    if (rc.qpn == 0) {
        rdma.reset();
        return;
    }
    if (rdma_activate(rc.qpn, remote_psn, rc.gid)) {
        rdma_qp_up = true;
        // T2 intent (both peers advertised Scope-B AND both opted in) keeps the
        // control plane on TCP, so the QP's completion queue carries only
        // WRITE_WITH_IMM. This boolean is symmetric on both ends — each side
        // computes (peer_wants AND local_wants) — so the transports never
        // diverge. T1 intent uses RDMA-SEND for the byte stream as before.
        const bool go_t2 = peer_scope_b_ && rpc_local_wants_scope_b();
        use_rdma = !go_t2;
        LOG_DBG("scope-b update_caps: peer_scope_b=%d local_want=%d go_t2=%d use_rdma=%d remote_psn=0x%x\n",
                (int) peer_scope_b_, (int) rpc_local_wants_scope_b(), (int) go_t2, (int) use_rdma, rc.psn);
    } else {
        GGML_LOG_ERROR("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#else
    (void)remote_caps;
#endif // GGML_RPC_RDMA
}

// --- Scope-B (T2) accessors --------------------------------------------------
bool socket_t::impl::rdma_activated() const {
#ifdef GGML_RPC_RDMA
    return rdma_qp_up;
#else
    return false;
#endif
}

bool socket_t::impl::peer_scope_b_caps() const {
#ifdef GGML_RPC_RDMA
    return peer_scope_b_;
#else
    return false;
#endif
}

bool socket_t::impl::scope_b_intent() const {
#ifdef GGML_RPC_RDMA
    return rdma_qp_up && peer_scope_b_ && rpc_local_wants_scope_b();
#else
    return false;
#endif
}

bool socket_t::impl::scope_b_setup(const rma_slot_config & cfg) {
#ifdef GGML_RPC_RDMA
    if (!rdma || !rdma_qp_up) {
        return false;
    }
    auto ch = std::make_unique<rma_channel_ibverbs>(rdma.get());
    if (!ch->register_slots(cfg)) {
        return false;
    }
    rma_chan = std::move(ch);
    // use_rdma is already false here (set in update_caps for T2 intent): the
    // control plane stays on TCP, the QP is dedicated to WRITE_WITH_IMM.
    GGML_LOG_INFO("Scope-B (T2) slots registered (control=TCP, data=RDMA WRITE_WITH_IMM)\n");
    return true;
#else
    (void) cfg;
    return false;
#endif
}

rma_channel * socket_t::impl::rma_get() const {
#ifdef GGML_RPC_RDMA
    return rma_chan.get();
#else
    return nullptr;
#endif
}

uint16_t socket_t::impl::rma_max_in_flight() const {
#ifdef GGML_RPC_RDMA
    return (uint16_t) RDMA_RX_DEPTH;
#else
    return 0;
#endif
}


/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

void socket_t::get_caps(uint8_t * local_caps) {
    return pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    return pimpl->update_caps(remote_caps);
}

bool socket_t::rdma_activated() const { return pimpl->rdma_activated(); }
bool socket_t::peer_scope_b() const { return pimpl->peer_scope_b_caps(); }
bool socket_t::scope_b_intent() const { return pimpl->scope_b_intent(); }
bool socket_t::setup_scope_b(const rma_slot_config & cfg) { return pimpl->scope_b_setup(cfg); }
rma_channel * socket_t::rma() const { return pimpl->rma_get(); }
uint16_t socket_t::rma_max_in_flight() const { return pimpl->rma_max_in_flight(); }
int socket_t::fd() const { return (int) pimpl->fd; }
bool socket_t::peer_closed() const {
#ifdef GGML_RPC_RDMA
    return pimpl->tcp_peer_closed();
#else
    // No RDMA busy-poll exists to unwedge on a TCP-only build; the blocking
    // recv path surfaces a closed peer directly.
    return false;
#endif
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_keepalive(sockfd_t sockfd) {
#ifndef _WIN32
    // Active liveness probing on the TCP CONTROL socket so a dead/half-open peer (e.g. a head killed
    // mid-stream) is detected even when the FIN/RST is missed (POLLRDHUP not compiled in, or a silent
    // half-open with no FIN): after KEEPIDLE idle seconds the kernel sends KEEPCNT probes KEEPINTVL
    // apart, and on no response the socket errors -> tcp_peer_closed()'s poll() sees POLLERR and the
    // blocking RDMA rdma_poll / TCP recv unwedges -> the relay closes the chain and re-accepts.
    // ~16s worst-case. The control socket is idle during RDMA data transfer, so probes actually run.
    int on = 1, idle = 10, intvl = 3, cnt = 2;
    setsockopt(sockfd, SOL_SOCKET,  SO_KEEPALIVE,  (char *)&on,    sizeof(on));
#ifdef TCP_KEEPIDLE
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE,  (char *)&idle,  sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, (char *)&intvl, sizeof(intvl));
#endif
#ifdef TCP_KEEPCNT
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT,   (char *)&cnt,   sizeof(cnt));
#endif
#endif
    return true;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    set_keepalive(client_socket_fd);   // dead-peer detection for the accepted (upstream) link
    return socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        GGML_LOG_ERROR("Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        GGML_LOG_ERROR("Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    set_keepalive(sockfd);   // dead-peer detection for the connected (downstream) link
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        GGML_LOG_ERROR("Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_rpc_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (!g_rpc_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}
