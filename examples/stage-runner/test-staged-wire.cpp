// test-staged-wire: golden-bytes regression for the STG2 forward-edge framing
// (examples/stage-runner/stage-wire-framing.h). Deterministic synthetic blob ->
// framing_send over a socketpair -> raw bytes captured -> byte-exact compare vs
// the checked-in golden (test-data/stg2-frame.golden), plus a framing_recv
// round-trip. Fails loudly on any framing change: these bytes ARE the wire
// contract between fork and internal ring stages (and what the RDMA packer must
// reproduce inside a slot).
//
// Self-contained: depends only on stage-wire-framing.h + libc/POSIX.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

#include "stage-wire-framing.h"

#ifndef STG2_GOLDEN_PATH
#define STG2_GOLDEN_PATH "test-data/stg2-frame.golden"   // cwd fallback (ctest WORKING_DIRECTORY)
#endif

static int mid_shape_violations = 0;

// ---- the fixed synthetic blob (deterministic; regenerate golden ONLY with a
// deliberate wire-protocol change, and then bump STG2 -> STG3 everywhere) ----
struct fixed_blob {                       // resize() mirrors production hidden_blob
    int32_t n_rows = 0, n_embd = 0;       // but asserts the shape never changes
    std::vector<int32_t> seq, pos;        // mid
    std::vector<float>   data;
    void resize(int32_t rows, int32_t embd) {
        if (n_rows && (rows != n_rows || embd != n_embd)) {
            fprintf(stderr, "FATAL: framing_recv re-shaped a mid-payload blob (%d,%d)->(%d,%d)\n",
                    n_rows, n_embd, rows, embd);
            ++mid_shape_violations;
        }
        n_rows = rows; n_embd = embd;
        seq.resize(rows); pos.resize(rows); data.resize((size_t) rows * embd);
    }
};

static void make_fixed(fixed_blob & b) {
    b.n_rows = 3; b.n_embd = 4;
    b.seq = { 7, 42, 99 };
    b.pos = { 0, 1, 2 };
    b.data.assign(3 * 4, 0.f);
    for (int r = 0; r < b.n_rows; ++r)
        for (int c = 0; c < b.n_embd; ++c)
            b.data[r * b.n_embd + c] = float(r * 10 + c) + 0.5f;
}

// capture framing_send over a socketpair into `out`
static bool capture(const fixed_blob & in, std::string & out, int n_bytes) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return false;
    if (!framing_send(sv[0], in)) { close(sv[0]); close(sv[1]); return false; }
    shutdown(sv[0], SHUT_WR);
    out.clear();
    char buf[4096];
    ssize_t k;
    while ((k = ::recv(sv[1], buf, sizeof(buf), 0)) > 0) out.append(buf, (size_t) k);
    close(sv[0]); close(sv[1]);
    return (int) out.size() == n_bytes;
}

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%-46s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };

    fixed_blob a; make_fixed(a);
    const int n_bytes = (int) framing_size(a.n_rows, a.n_embd);
    // 12 hdr + 2*12 seq/pos + 48 fp32 = 84
    check(n_bytes == 84, "framing_size matches layout");

    // 1) golden byte-exactness
    std::string got;
    bool cap_ok = capture(a, got, n_bytes);
    check(cap_ok, "capture all frame bytes over socketpair");
    FILE * f = fopen(STG2_GOLDEN_PATH, "rb");
    check(f != nullptr, "golden file opens (" STG2_GOLDEN_PATH ")");
    if (f) {
        std::string want;
        char buf[4096]; size_t k;
        while ((k = fread(buf, 1, sizeof(buf), f)) > 0) want.append(buf, k);
        fclose(f);
        bool same = want == got;
        if (!same) {
            printf("  golden %zu bytes vs captured %zu bytes, first diff at:",
                   want.size(), got.size());
            size_t i = 0, n = std::min(want.size(), got.size());
            for (; i < n; ++i) if (want[i] != got[i]) { printf(" %zu: %02x != %02x\n", i, (uint8_t) want[i], (uint8_t) got[i]); break; }
            if (i == n) printf(" length only\n");
        }
        check(same, "wire bytes == checked-in golden");
    } else ++fails;

    // 2) round-trip through framing_recv
    {
        int sv[2];
        bool rt = socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0;
        if (rt) {
            fixed_blob w; make_fixed(w);
            rt = framing_send(sv[0], w);
            fixed_blob r{ 0, 0, {}, {}, {} };
            if (rt) rt = framing_recv(sv[1], r);
            rt = rt && r.n_rows == w.n_rows && r.n_embd == w.n_embd &&
                 r.seq == w.seq && r.pos == w.pos && r.data == w.data;
            close(sv[0]); close(sv[1]);
        }
        check(rt, "framing_send -> framing_recv round-trip");
    }

    // 3) magic mismatch is rejected (a non-STG2 stream must not parse)
    {
        int sv[2];
        bool rej = socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0;
        if (rej) {
            const char junk[12] = { 'X', 'Y', 'Z', '1', 0, 0, 0, 3, 0, 0, 0, 4 };
            ssize_t k = ::send(sv[0], junk, sizeof(junk), 0); (void) k;
            shutdown(sv[0], SHUT_WR);
            fixed_blob r{ 0, 0, {}, {}, {} };
            rej = !framing_recv(sv[1], r);          // must refuse
            close(sv[0]); close(sv[1]);
        }
        check(rej, "non-STG2 magic rejected");
    }

    check(mid_shape_violations == 0, "recv never re-shapes a blob mid-payload");

    printf("%s (%d failures)\n", fails ? "TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails ? 1 : 0;
}
