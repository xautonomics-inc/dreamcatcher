#pragma once
// STG2 hidden-blob framing on the stage forward edge — the SINGLE definition of
// the byte layout, shared by stage-runner.cpp (production) and
// test-staged-wire.cpp (golden-bytes regression). Fork<->internal ring interop
// depends on these bytes: any change here is a wire-protocol change.
//
// Layout (all little-endian, no padding):
//   int32  magic   = 0x53544732 ("STG2", v2 = per-row tagged)
//   int32  n_rows
//   int32  n_embd
//   int32  seq[n_rows]
//   int32  pos[n_rows]
//   float  data[n_rows * n_embd]   (row-major fp32)
//
// The RDMA forward path (STAGE_RDMA_TRANSPORT) packs EXACTLY these bytes into a
// slot via pack_hidden(); on the TCP edge (always, and always on a non-RDMA
// negotiated conn) framing_send/framing_recv put them on the socket verbatim.
// Golden: examples/stage-runner/test-data/stg2-frame.golden.

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <sys/socket.h>
#include <unistd.h>

static const int32_t STAGE_WIRE_MAGIC = 0x53544732;  // "STG2"

struct stage_wire_blob {
    int32_t n_rows = 0;
    int32_t n_embd = 0;
    int32_t * seq = nullptr;   // n_rows entries
    int32_t * pos = nullptr;   // n_rows entries
    const float * data = nullptr;  // n_rows * n_embd entries (send side)
};

inline bool stage_all_send(int fd, const void * b, size_t n) {
    const char * p = (const char *) b;
    while (n) { ssize_t k = ::send(fd, p, n, 0); if (k <= 0) return false; p += k; n -= (size_t) k; }
    return true;
}
inline bool stage_all_recv(int fd, void * b, size_t n) {
    char * p = (char *) b;
    while (n) { ssize_t k = ::recv(fd, p, n, 0); if (k <= 0) return false; p += k; n -= (size_t) k; }
    return true;
}

// Write one STG2 frame. Blob types are duck-typed on {n_rows, n_embd, seq, pos,
// data}: production hidden_blob (std::vector members) and the test's fixed-size
// buffers both fit. .data() yields raw pointers for both; plain pointers work via
// the Blob && binding when the type has no .data().
template <typename Blob>
inline bool framing_send(int fd, const Blob & v) {
    const int32_t hdr[3] = { STAGE_WIRE_MAGIC, v.n_rows, v.n_embd };
    return stage_all_send(fd, hdr, sizeof(hdr)) &&
           stage_all_send(fd, v.seq.data(),  (size_t) v.n_rows * sizeof(int32_t)) &&
           stage_all_send(fd, v.pos.data(),  (size_t) v.n_rows * sizeof(int32_t)) &&
           stage_all_send(fd, v.data.data(), (size_t) v.n_rows * v.n_embd * sizeof(float));
}

// Read one STG2 frame. On true, (seq/pos/data) hold exactly the sent values;
// magic mismatch or truncated frame -> false. Blobs exposing resize(rows, embd)
// (production hidden_blob) are sized from the header before the payload is read;
// fixed-buffer blobs (tests) must be pre-sized by the caller.
namespace framing_detail {
    template <typename C, typename A> static auto resize_test(int)
        -> decltype(std::declval<C &>().resize(std::declval<A>(), std::declval<A>()), std::true_type {});
    template <typename C, typename A> static std::false_type resize_test(...);
    template <typename C, typename A>
    using resize_v = decltype(resize_test<C, A>(0));
}
template <typename Blob>
inline void framing_maybe_resize(Blob & v, int32_t rows, int32_t embd) {
    if constexpr (framing_detail::resize_v<Blob, int32_t>::value) v.resize(rows, embd);
    (void) rows; (void) embd;
}
template <typename Blob>
inline bool framing_recv(int fd, Blob & v) {
    int32_t hdr[3];
    if (!stage_all_recv(fd, hdr, sizeof(hdr)) || hdr[0] != STAGE_WIRE_MAGIC) return false;
    framing_maybe_resize(v, hdr[1], hdr[2]);
    v.n_rows = hdr[1]; v.n_embd = hdr[2];
    return stage_all_recv(fd, v.seq.data(), (size_t) v.n_rows * sizeof(int32_t)) &&
           stage_all_recv(fd, v.pos.data(), (size_t) v.n_rows * sizeof(int32_t)) &&
           stage_all_recv(fd, v.data.data(), (size_t) v.n_rows * v.n_embd * sizeof(float));
}

// Byte size of a frame's on-wire representation (used by tests and the RDMA packer).
inline size_t framing_size(int32_t n_rows, int32_t n_embd) {
    return 3 * sizeof(int32_t) + 2 * (size_t) n_rows * sizeof(int32_t)
         + (size_t) n_rows * n_embd * sizeof(float);
}
