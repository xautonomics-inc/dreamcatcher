// gslot_client.h -- header-only C++17 client for the gslot arbiter.
//
// Drop this file next to stage-runner.cpp and #include it.  It needs no CMake
// change (stage-server.h is included the same way and is not listed either) and
// no library beyond POSIX sockets.
//
// USAGE in the wave-scheduler pump, at the single dispatch gate:
//
//     static gslot::gate g_gslot;                       // file scope
//     ...
//     const bool gate_open = (!wave_stagger || now >= t_next)
//                            && g_gslot.open(now);      // <-- the whole change
//
// DEFAULT OFF.  With STAGE_GSLOT_SOCKET unset, open() returns true after one
// predictable-branch load of a bool -- the v1 path stays byte-identical in
// behaviour, which is the same discipline STAGE_WAVE_SCHED shipped under.
//
// TWO LEASE SHAPES.  Select with STAGE_GSLOT_MODE.
//
// `quantum` (default) -- take ONE lease per quantum (default 250 ms) and answer
// locally until it expires.  One round trip per quantum per tenant, ~0.04% of
// the quantum.  Right when two tenants each want CONTINUOUS throughput on one
// device: they alternate in coarse runs instead of interleaving per kernel.
//
// `burst` -- acquire before this stage computes, release the instant the wave
// is handed off.  Right for a PIPELINE stage, which computes for a small slice
// of each token and then waits for the ring to come round: at S0-measured busy
// fractions every ring stage but the bottleneck is idle >90% of every token,
// and a quantum hold would sit on the device through all of it, starving the
// co-tenant AND leaving the resource idle.  Under `burst` both tenants stay
// sized FULL WIDTH -- no affinity changes, no threadpool resizing, nothing in
// the hot path -- and the lease decides *who computes*, not *how wide*.
// Exclusivity in time instead of partitioning in space.
//
// The cost of `burst` is one round trip per wave instead of per quantum, so it
// is measured: `acquire_us_total`/`granted` is reported at /admin/gslot.
//
// FAIL-OPEN, ALWAYS.  Unreachable socket, timeout, malformed reply, arbiter
// restart: every one of them returns "permitted" and disables the client for a
// backoff window.  A scheduler that can stop the ring by dying is worse than no
// scheduler.  There is deliberately no configuration option to make this fail
// closed.

#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

namespace gslot {

inline double now_ms_monotonic() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

class gate {
public:
    gate() {
        const char * sock = getenv("STAGE_GSLOT_SOCKET");
        if (!sock || !*sock) { return; }               // default OFF
        sock_path_ = sock;
        const char * t = getenv("STAGE_GSLOT_TENANT");
        tenant_ = (t && *t) ? t : default_tenant();
        const char * r = getenv("STAGE_GSLOT_RESOURCE");
        resource_ = (r && *r) ? r : "cpu:host";
        quantum_ms_ = [] {
            const char * e = getenv("STAGE_GSLOT_QUANTUM_MS");
            int v = e ? atoi(e) : 250;
            return v < 10 ? 10 : (v > 5000 ? 5000 : v);
        }();
        retry_ms_ = [] {
            const char * e = getenv("STAGE_GSLOT_RETRY_MS");
            int v = e ? atoi(e) : 5;
            return v < 1 ? 1 : (v > 1000 ? 1000 : v);
        }();
        weight_ = [] {
            const char * e = getenv("STAGE_GSLOT_WEIGHT");
            double v = e ? atof(e) : 1.0;
            return v <= 0.0 ? 1.0 : v;
        }();
        const char * m = getenv("STAGE_GSLOT_MODE");
        burst_ = (m && strcmp(m, "burst") == 0);
        enabled_ = true;
    }

    ~gate() { disconnect(); }

    gate(const gate &)             = delete;
    gate & operator=(const gate &) = delete;

    bool active() const { return enabled_; }

    // The gate.  Returns true when this tenant may dispatch compute right now.
    // `now` is milliseconds on any monotonic base; pass the pump's own clock so
    // the two agree.
    bool open(double now) {
        if (!enabled_) { return true; }
        // BURST: hold only while actually computing. No quantum, no local
        // expiry -- handoff() gives the device back the moment the wave leaves.
        if (burst_) {
            if (holding_) { return true; }
            if (now < next_try_ms_) { return false; }
            heartbeat_if_due(now);
            if (!ensure_registered()) { fail_open(now); return true; }
            const double t0 = now_ms_monotonic();
            long l = do_lease();
            acquire_us_total_ += (unsigned long long) ((now_ms_monotonic() - t0) * 1000.0);
            if (l < 0) { fail_open(now); return true; }
            if (l == 0) { next_try_ms_ = now + retry_ms_; blocked_++; return false; }
            lease_id_ = l; holding_ = true; held_since_ = now_ms_monotonic(); granted_++;
            return true;
        }
        if (holding_ && now < hold_until_) { return true; }
        if (holding_) { do_release(); }
        if (now < next_try_ms_) { return false; }
        heartbeat_if_due(now);
        if (!ensure_registered()) { fail_open(now); return true; }
        long lease = do_lease();
        if (lease < 0) {                       // transport fault -> fail open
            fail_open(now);
            return true;
        }
        if (lease == 0) {                      // arbiter says: someone else's turn
            next_try_ms_ = now + retry_ms_;
            blocked_++;
            return false;
        }
        lease_id_    = lease;
        holding_     = true;
        hold_until_  = now + quantum_ms_;
        granted_++;
        return true;
    }

    // Call when the tenant goes idle so the turn is handed back early rather
    // than burning the rest of the quantum on nothing.
    void yield() {
        if (holding_) { do_release(); }
    }

    // BURST mode: the wave has left this stage; the device is free. Calling
    // this is what turns a lease into idle-window harvesting rather than a
    // hold. A no-op in quantum mode and when the gate is off.
    void handoff() {
        if (!enabled_ || !burst_ || !holding_) { return; }
        do_release();
    }

    // Convenience for a pump loop: `want` is false when there is nothing to
    // dispatch, in which case the turn is handed back immediately instead of
    // burning the rest of the quantum on an idle slot.  Returns whether this
    // tenant may dispatch now.  (NOT named gate(): a member function with the
    // class's own name is a constructor declaration.)
    bool may_dispatch(double now, bool want) {
        if (!want) { yield(); return true; }
        return open(now);
    }

    unsigned long granted() const { return granted_; }
    unsigned long blocked() const { return blocked_; }
    unsigned long faults()  const { return faults_;  }
    bool          burst()   const { return burst_;   }
    unsigned long long acquire_us_total() const { return acquire_us_total_; }
    double        held_ms_total()  const { return held_ms_total_; }
    const std::string & tenant() const { return tenant_; }

private:
    static std::string default_tenant() {
        char host[128] = {0};
        gethostname(host, sizeof(host) - 1);
        char buf[192];
        snprintf(buf, sizeof(buf), "stage-runner@%s:%d", host, (int) getpid());
        return std::string(buf);
    }

    void fail_open(double now) {
        faults_++;
        disconnect();
        registered_ = false;
        holding_    = false;
        // Back off hard.  A flapping arbiter must not turn into an RPC storm on
        // the pump thread, which is the one thread that owns all ring I/O.
        next_try_ms_ = now + 2000.0;
    }

    void disconnect() {
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
        rx_.clear();
    }

    bool ensure_connected() {
        if (fd_ >= 0) { return true; }
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { return false; }
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path_.c_str());
        if (connect(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) { close(fd); return false; }
        struct timeval tv;
        tv.tv_sec  = 0;
        tv.tv_usec = 20000;   // 20 ms: the pump must never stall on this
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        fd_ = fd;
        rx_.clear();
        return true;
    }

    bool ensure_registered() {
        if (registered_ && fd_ >= 0) { return true; }
        if (!ensure_connected()) { return false; }
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "{\"id\":%d,\"op\":\"register\",\"tenant\":\"%s\",\"resource\":\"%s\","
                 "\"mode\":\"turn\",\"weight\":%.4f,\"pid\":%d}\n",
                 next_id(), tenant_.c_str(), resource_.c_str(), weight_, (int) getpid());
        std::string reply;
        if (!rpc(msg, reply)) { return false; }
        // "already registered" is fine: a reconnect after an arbiter restart.
        registered_ = has_true(reply, "\"ok\"") || reply.find("already registered") != std::string::npos;
        return registered_;
    }

    // returns lease id (>0), 0 == busy, -1 == transport fault
    long do_lease() {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "{\"id\":%d,\"op\":\"lease\",\"tenant\":\"%s\",\"est_ms\":%d,\"nowait\":true}\n",
                 next_id(), tenant_.c_str(), quantum_ms_);
        std::string reply;
        if (!rpc(msg, reply)) { return -1; }
        if (!has_true(reply, "\"ok\"")) { return 0; }
        long v = extract_long(reply, "\"lease\":");
        return v > 0 ? v : 0;
    }

    void do_release() {
        if (holding_ && held_since_ > 0.0) {
            held_ms_total_ += now_ms_monotonic() - held_since_;
            held_since_ = 0.0;
        }
        holding_ = false;
        if (lease_id_ <= 0) { return; }
        char msg[192];
        snprintf(msg, sizeof(msg), "{\"id\":%d,\"op\":\"release\",\"lease\":%ld}\n",
                 next_id(), lease_id_);
        std::string reply;
        rpc(msg, reply);   // best effort; the arbiter expires it anyway
        lease_id_ = 0;
    }

    void heartbeat_if_due(double now) {
        if (now - last_hb_ms_ < 2000.0) { return; }
        last_hb_ms_ = now;
        if (!registered_ || fd_ < 0) { return; }
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "{\"id\":%d,\"op\":\"hb\",\"tenant\":\"%s\",\"progress\":%lu}\n",
                 next_id(), tenant_.c_str(), granted_);
        std::string reply;
        rpc(msg, reply);
    }

    int next_id() { return ++id_; }

    // One request, one reply.  Server pushes (id 0) are skipped.
    bool rpc(const char * msg, std::string & reply) {
        if (!ensure_connected()) { return false; }
        size_t len = strlen(msg), off = 0;
        while (off < len) {
            ssize_t n = send(fd_, msg + off, len - off, MSG_NOSIGNAL);
            if (n <= 0) { disconnect(); return false; }
            off += (size_t) n;
        }
        for (int guard = 0; guard < 8; guard++) {
            if (!read_line(reply)) { disconnect(); return false; }
            if (reply.find("\"op\":\"grant\"") == std::string::npos) { return true; }
        }
        disconnect();
        return false;
    }

    bool read_line(std::string & out) {
        for (;;) {
            size_t nl = rx_.find('\n');
            if (nl != std::string::npos) {
                out = rx_.substr(0, nl);
                rx_.erase(0, nl + 1);
                return true;
            }
            char buf[4096];
            ssize_t n = recv(fd_, buf, sizeof(buf), 0);
            if (n <= 0) { return false; }
            rx_.append(buf, (size_t) n);
            if (rx_.size() > (1u << 20)) { return false; }
        }
    }

    static bool has_true(const std::string & s, const char * key) {
        size_t p = s.find(key);
        if (p == std::string::npos) { return false; }
        p = s.find(':', p);
        if (p == std::string::npos) { return false; }
        while (++p < s.size() && (s[p] == ' ')) { }
        return s.compare(p, 4, "true") == 0;
    }

    static long extract_long(const std::string & s, const char * key) {
        size_t p = s.find(key);
        if (p == std::string::npos) { return -1; }
        return strtol(s.c_str() + p + strlen(key), nullptr, 10);
    }

    bool        enabled_    = false;
    bool        registered_ = false;
    bool        holding_    = false;
    bool        burst_      = false;
    int         fd_         = -1;
    int         id_         = 0;
    int         quantum_ms_ = 250;
    int         retry_ms_   = 5;
    double      weight_     = 1.0;
    long        lease_id_   = 0;
    double      hold_until_ = 0.0;
    double      next_try_ms_ = 0.0;
    double      last_hb_ms_  = 0.0;
    double      held_since_  = 0.0;
    double      held_ms_total_ = 0.0;
    unsigned long long acquire_us_total_ = 0;
    unsigned long granted_ = 0, blocked_ = 0, faults_ = 0;
    std::string sock_path_, tenant_, resource_, rx_;
};

}  // namespace gslot
