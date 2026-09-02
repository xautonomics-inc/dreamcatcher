#pragma once

// -----------------------------------------------------------------------------
// Scope-B RMA data plane: one-sided RDMA WRITE into pre-registered slot tables.
//
// This header is the *substrate-neutral* interface described in
// SPEC §B.2. The control plane and bootstrap stay on socket_t (transport.h);
// this trait adds a second, narrow data-plane channel co-owned by the same
// connection. An ibverbs implementation lives in transport.cpp (it reuses the
// connection's existing RC QP); a libfabric implementation is a documented
// future seam (SPEC §0/§B.2, Appendix B).
//
// The hot path (master->worker tensor delivery) becomes a single
// IBV_WR_RDMA_WRITE_WITH_IMM into a remote slot, self-signaled by a 32-bit
// immediate, with no per-tensor request/response round-trip.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// --- Scope-B capability negotiation (SPEC §C.6) ------------------------------
// The 24-byte conn_caps carries rdma_caps{qpn, psn, gid[16]}. PSN is 24-bit and
// stored in a 32-bit field, so the high bits are reserved-zero on the wire. We
// claim bit 31 of psn = "this peer can do Scope-B (T2)". An older peer ignores
// it (psn is masked to 24 bits in rdma_activate), so it transparently stays on
// T1/T0. Both sides must set it (and both must opt in via GGML_RPC_PROTOCOL=v2)
// for the slot-table handshake to proceed.
static constexpr uint32_t RPC_RMA_CAP_SCOPE_B = 1u << 31;
static constexpr uint32_t RPC_RMA_PSN_MASK    = 0x00ffffffu; // PSN is 24-bit

// --- Default slot table geometry (SPEC §C.1) ---------------------------------
// Two size classes, host-pinned (not VRAM: GPUDirect is out of scope), sized
// from the model graph in principle; these are the first-guess defaults the
// microbench (#26) is meant to tune. Decode slots carry per-step activations;
// bulk slots carry larger prefill / graph blobs.
struct rma_slot_config {
    uint32_t decode_slot_size = 256u * 1024u;       // 256 KiB
    uint16_t n_decode         = 64;                 // 64 decode slots
    uint32_t bulk_slot_size   = 8u * 1024u * 1024u; // 8 MiB
    uint16_t n_bulk           = 4;                  // 4 bulk slots

    uint16_t n_slots()  const { return (uint16_t)(n_decode + n_bulk); }
    bool     is_bulk(uint16_t i) const { return i >= n_decode; }
    uint32_t slot_size(uint16_t i) const { return is_bulk(i) ? bulk_slot_size : decode_slot_size; }
    // byte offset of slot i within a single contiguous region (decode block
    // first, then bulk block), so the whole table is one MR / one rkey.
    uint64_t slot_offset(uint16_t i) const {
        if (!is_bulk(i)) {
            return (uint64_t)i * decode_slot_size;
        }
        return (uint64_t)n_decode * decode_slot_size +
               (uint64_t)(i - n_decode) * bulk_slot_size;
    }
    uint64_t total_size() const {
        return (uint64_t)n_decode * decode_slot_size + (uint64_t)n_bulk * bulk_slot_size;
    }
};

// --- A remote-writable slot advertised to the peer (SPEC §B.2) ---------------
// Exchanged at handshake via the slot-announce control message. #pragma pack so
// it is wire-safe (it travels inside the control-plane message body,
// little-endian).
#pragma pack(push, 1)
struct rma_region {
    uint64_t addr;     // peer virtual address (ibv: VA; libfabric: FI_MR_VIRT_ADDR)
    uint64_t rkey;     // remote key (ibv rkey; libfabric fi_mr_key)
    uint32_t len;      // slot capacity in bytes
    uint16_t slot_idx; // index in the peer's slot table
    uint16_t _pad;     // keep 8-byte alignment / deterministic size
};
#pragma pack(pop)
static_assert(sizeof(rma_region) == 24, "rma_region must be 24 bytes on the wire");

// A drained completion: which slot was written (via imm) and how many bytes.
struct rma_completion {
    uint32_t imm;      // decoded immediate: (slot_idx << 16) | seq
    uint32_t byte_len; // total bytes written into the slot ([desc|offset|data])
};

// --- Immediate (imm_data) encoding (SPEC §C.5) -------------------------------
// imm_data = (slot_idx << 16) | (seq & 0xFFFF). Sent in network byte order on
// the wire by ibverbs; decoded on the receiver via ntohl.
static inline uint32_t rma_pack_imm(uint16_t slot_idx, uint16_t seq) {
    return ((uint32_t)slot_idx << 16) | (uint32_t)seq;
}
static inline uint16_t rma_imm_slot(uint32_t imm) { return (uint16_t)(imm >> 16); }
static inline uint16_t rma_imm_seq (uint32_t imm) { return (uint16_t)(imm & 0xFFFF); }

// -----------------------------------------------------------------------------
// slot_table: substrate-neutral free-list bookkeeping over a peer's decode
// slots (SPEC §C.2). Used master-side to pick a free remote slot to WRITE into
// and to release it once the worker has consumed it. Header-only + fabric-free
// so it is unit-testable with no RDMA. Bulk slots are managed separately
// (chunked WRITE — future), so this tracks only the decode class.
// -----------------------------------------------------------------------------
struct slot_table {
    void init(uint16_t n_decode) {
        n_decode_ = n_decode;
        release_all();
    }

    // Acquire a free decode slot, honoring the in-flight cap (bounded by the
    // peer's posted receive ring). Returns -1 if none available (caller then
    // falls back to the T1/T0 byte stream — SPEC §C.2.3).
    int acquire(uint16_t max_in_flight) {
        if (free_list.empty() || in_flight_ >= max_in_flight) {
            return -1;
        }
        uint16_t s = free_list.back();
        free_list.pop_back();
        in_flight_++;
        return s;
    }

    void release(uint16_t slot) {
        free_list.push_back(slot);
        if (in_flight_ > 0) {
            in_flight_--;
        }
    }

    // Release every outstanding slot at once. The prototype uses the GET_TENSOR
    // readback as a per-step barrier: once the worker has computed the step, all
    // decode slots it consumed are free again.
    void release_all() {
        free_list.clear();
        free_list.reserve(n_decode_);
        for (int i = n_decode_ - 1; i >= 0; --i) {
            free_list.push_back((uint16_t)i);
        }
        in_flight_ = 0;
    }

    uint16_t in_flight() const { return in_flight_; }
    uint16_t available() const { return (uint16_t)free_list.size(); }

private:
    std::vector<uint16_t> free_list;
    uint16_t              n_decode_  = 0;
    uint16_t              in_flight_ = 0;
};

// -----------------------------------------------------------------------------
// rma_channel: one per RDMA-capable connection. Substrate-swappable.
// -----------------------------------------------------------------------------
struct rma_channel {
    virtual ~rma_channel() = default;

    // ---- setup (control-plane assisted; runs once at handshake) ----
    // Allocate + pin + register the local landing slots described by cfg.
    virtual bool register_slots(const rma_slot_config & cfg) = 0;
    // {addr,rkey,len,i} for local slot i, to advertise to the peer.
    virtual rma_region local_slot(uint16_t i) const = 0;
    virtual uint16_t   n_slots() const = 0;
    // Record the peer's advertised slot table (the WRITE targets).
    virtual void       set_remote_slots(const rma_region * slots, uint16_t n) = 0;

    // Local staging/landing pointer + capacity for slot i. The initiator
    // assembles [rpc_tensor | offset | data] into slot_ptr(i) (inside the
    // registered region) before calling write_slot with that pointer.
    virtual uint8_t *  slot_ptr(uint16_t i) = 0;
    virtual uint32_t   slot_capacity(uint16_t i) const = 0;

    // ---- hot path (initiator side) ----
    // One-sided WRITE of [buf, len] into the peer's remote_slot, self-signaled
    // via the 32-bit immediate. buf must lie within this channel's registered
    // region (e.g. a value returned by slot_ptr). Blocks only on the local send
    // completion (data delivered to peer memory), never on the peer.
    virtual bool write_slot(uint16_t remote_slot, const void * buf, uint32_t len,
                            uint32_t imm) = 0;

    // ---- hot path (target side) ----
    // Non-blocking drain: pull up to `max` completed WRITE_WITH_IMM events into
    // `out`. Returns count (0..max), or <0 on error. Re-posts consumed recvs.
    virtual int  poll_writes(rma_completion * out, int max) = 0;

    virtual void shutdown() = 0;

    // ---- master-side bookkeeping (unused on the pure target) ----
    // Carried here because it is per-connection state with the channel's
    // lifetime. The initiator stages into local slot i and WRITEs to the paired
    // remote slot i (identical geometry both ends), tracking i via this free
    // list and stamping each WRITE with next_seq.
    slot_table master_slots;
    uint16_t   next_seq = 0;
};
