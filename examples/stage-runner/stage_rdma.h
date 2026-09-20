#pragma once
// Opaque RoCEv2 RDMA (Scope-B) transport for the stage-runner forward path.
// Self-contained TU (stage_rdma.cpp includes transport.h); this header exposes
// NO socket_t / ggml-rpc types, so stage-runner.cpp can also include httplib
// (which has its own `using socket_t = int`) without a collision.
#include <cstdint>
#include <cstddef>

struct stage_conn;   // opaque

// Establish a stage link. If STAGE_RDMA=1 and the peer also negotiates Scope-B
// (GGML_RPC_PROTOCOL=v2), the forward path becomes one-sided RDMA WRITE; else it
// transparently stays on the TCP byte stream. host=nullptr => listen, else connect.
stage_conn* stage_conn_connect(const char* host, int port);
stage_conn* stage_conn_listen(int port);
bool        stage_conn_ok(stage_conn*);
bool        stage_conn_is_rdma(stage_conn*);     // forward path is RDMA
int         stage_conn_fd(stage_conn*);          // raw TCP control fd (back-edge + shutdown)
void        stage_conn_close(stage_conn*);

// Control / back-edge byte stream (always TCP; the QP is dedicated to WRITE).
bool        stage_conn_send(stage_conn*, const void* buf, size_t len);
bool        stage_conn_recv(stage_conn*, void* buf, size_t len);

// Forward hot path. write: one-sided WRITE of [buf,len] into a remote slot
// (len must be <= a decode slot; caller falls back to *_send for oversize).
// read: blocking poll for the next inbound WRITE -> returns landing ptr + *len,
// valid until the next stage_conn_read. Both return false/nullptr if !is_rdma.
bool         stage_conn_write(stage_conn*, const void* buf, uint32_t len);
const void*  stage_conn_read (stage_conn*, uint32_t* len);
uint32_t     stage_conn_slot_cap(stage_conn*);   // max bytes for stage_conn_write
