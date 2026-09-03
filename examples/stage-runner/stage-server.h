// OpenAI-compatible server: ik-llama-stage-runner-as-head driving a multi-stage MTP ring.
// Ported from the mainline fork's stage-server.h onto the ik_llama stage driver: included by
// stage-runner.cpp before main() so it sees model_bundle, run_tokens, send_hidden, recv_mtp_msg,
// tcp_connect, tcp_listen_accept, mtp_msg, etc. Compiled only under STAGE_SERVER.
//
// Default speed mode = token-back MTP (the tail samples + drafts). slots=1, synchronous depth-1.
//
// STATEFUL KV CACHE (for 256K agent sessions): the ring connection is PERSISTENT and its KV is kept
// across requests. Each request is prefix-matched against the cached token sequence; only the NEW
// (suffix) tokens are prefilled (append path -> no reconnect, no full re-prefill). The ring is
// reconnected (which clears every stage's KV via its per-connection reset) ONLY when there is no
// connection yet or the new prompt diverges from the cache (rare for a linear agent). logits-back
// (--logits-back) is a planned alt mode; the flag is wired and currently falls back to token-back.
#pragma once
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <functional>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <deque>
#include <cerrno>
#include <cstring>

static std::string sv_detok(model_bundle & b, int tok) {
    char buf[256];
    int n = llama_token_to_piece_vocab(b.vocab, (llama_token) tok, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

static std::vector<llama_token> sv_tokenize(model_bundle & b, const std::string & s, bool add_bos) {
    int n = -llama_vocab_tokenize(b.vocab, s.c_str(), s.size(), nullptr, 0, add_bos, true);
    if (n < 0) n = 0;
    std::vector<llama_token> t(n);
    if (n) llama_vocab_tokenize(b.vocab, s.c_str(), s.size(), t.data(), t.size(), add_bos, true);
    return t;
}

static std::string sv_apply_chat_template(model_bundle & b, const nlohmann::json & messages) {
    std::vector<std::string> roles, contents;
    for (auto & mm : messages) { roles.push_back(mm.value("role", std::string("user")));
                                 contents.push_back(mm.value("content", std::string(""))); }
    std::vector<llama_chat_message> msgs;
    for (size_t i = 0; i < roles.size(); ++i) msgs.push_back({ roles[i].c_str(), contents[i].c_str() });
    const char * tmpl = llama_model_chat_template(b.model, nullptr);
    if (tmpl) {
        int need = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, nullptr, 0);
        if (need > 0) { std::string out(need, '\0');
            int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, out.data(), out.size());
            if (n > 0) { out.resize(n); return out; } }
    }
    std::string out;
    for (size_t i = 0; i < roles.size(); ++i) out += roles[i] + ": " + contents[i] + "\n";
    out += "assistant:";
    return out;
}

// ============================ MULTI-SLOT PIPELINED SERVER ============================
// Staggers up to `n_slots` INDEPENDENT requests through the ring so different pipeline stages
// work on different requests at the same instant (pipeline parallelism). The synchronous gen()
// below processes one wave end-to-end per token (~full-ring latency) with a single stage busy at
// a time; here each active request keeps ONE wave in flight, so aggregate throughput scales with
// the number of concurrent requests up to the pipeline depth (#stages). --slots 1 (default) keeps
// the proven synchronous prefix-cached single-slot path (gen()) untouched.
//
// CORRECTNESS MODEL
//  * Fresh seq per job. Every request is bound to a NEW monotonically-increasing seq id (KV
//    namespace) in [0, n_seq_max). A never-before-used seq has no stale KV on ANY stage, so no
//    per-seq reset is needed -- which matters because the relays have NO per-seq KV clear (only a
//    full reconnect wipes them). Seq ids are recycled only by a full ring reconnect, performed
//    ONLY when the ring is fully idle (no wave in flight, every slot free). KV stays bounded:
//    n_seq_max partitions of (n_ctx/n_seq_max) each.
//  * FIFO demux. The return edge (mtp_msg) carries no seq, so returns are matched to slots by a
//    strict FIFO queue: the ring preserves wave order end-to-end and the tail emits exactly one
//    mtp_msg per received wave (real or dropped-filler), so pop-front == the oldest outstanding
//    wave's slot. No wire change, no relay/tail code change.
//  * Chunked prefill. A prompt is split into waves (<= pf_wave, never exactly k+1 rows so the tail
//    never misreads one as a verify wave). The tail runs its prefill branch for each; only the
//    FINAL prefill response is the first generated token (earlier ones are mid-prompt argmaxes,
//    discarded). One in-flight wave per decoding slot -> aggregate in-flight == #active slots.
//
// DEPLOYMENT (all mandatory, else silent corruption / drops):
//  * Every stage (head + relays + tail) launched with --n-seq-max >= n_slots so each seq's ctx
//    partition (n_ctx/n_seq_max) holds its prompt+gen. (Default-64 relays cap a seq at 256!)
//  * Tail launched with STAGE_N_REAL >= n_slots, else seqs >= g_n_real are dropped as fillers.
//  * httplib worker-thread pool >= n_slots (each in-flight request blocks one worker draining its
//    token queue).
static void run_server_pipelined(model_bundle & b, const std::string & connect_to, int return_listen,
                                 int http_port, int default_max_tokens, const std::string & model_id,
                                 int n_slots, int n_seq_max) {
    using json = nlohmann::json;
    const int n_ctx  = (int) llama_n_ctx(b.ctx);
    const int mtp_k  = []{ const char * e = getenv("STAGE_MTP_NDRAFT"); int k = e ? atoi(e) : 1; return k < 1 ? 1 : (k > 7 ? 7 : k); }();
    const int pf_wave= []{ const char * e = getenv("STAGE_PREFILL_WAVE"); int w = e ? atoi(e) : 256; return w < 1 ? 256 : w; }();
    const int PF_DEPTH = []{ const char * e = getenv("STAGE_PF_DEPTH"); int d = e ? atoi(e) : 4; return d < 1 ? 1 : d; }();
    if (n_seq_max < n_slots) n_seq_max = n_slots;
    const int per_seq_ctx = n_ctx / (n_seq_max > 0 ? n_seq_max : 1);
    fprintf(stderr, "server[pipe]: slots=%d seq_budget=%d k=%d pf_wave=%d n_ctx=%d per_seq_ctx=%d. OpenAI API on http://0.0.0.0:%d\n",
            n_slots, n_seq_max, mtp_k, pf_wave, n_ctx, per_seq_ctx, http_port);

    struct PipeJob {
        std::vector<int32_t> prompt; int max_tok = 0;
        std::mutex mtx; std::condition_variable cv;
        std::deque<int32_t> toks;      // emitted tokens (FIFO), drained by the HTTP worker
        bool started = false, done = false, failed = false;
    };
    enum SlotPhase { FREE = 0, PREFILL = 1, DECODE = 2 };
    struct PipeSlot {
        int seq = -1; int phase = FREE;
        std::shared_ptr<PipeJob> job;
        std::vector<std::vector<int32_t>> pf_tok;  // prefill wave token slices
        std::vector<int32_t> pf_base;              // prefill wave base positions
        size_t pf_sent = 0, pf_recv = 0;           // waves handed to fwd / responses seen
        int    inflight = 0;                       // waves of THIS slot currently in the ring
        std::vector<int32_t> nx_issue; int32_t nx_base = 0; bool decode_ready = false;
        int    produced = 0, max_tok = 0;
    };

    std::mutex M; std::condition_variable CV;      // guards all scheduler state; wakes the fwd thread
    std::deque<int>  inflight_slot;                // slot per in-flight wave (FIFO, for bwd demux)
    std::vector<PipeSlot> slots(n_slots);
    std::deque<std::shared_ptr<PipeJob>> pending;
    int  next_seq = 0;                             // fresh-seq allocator (recycled only on idle reconnect)
    int  fd = -1, rfd = -1;
    bool stop = false;

    auto ring_close = [&]() {
        if (fd  >= 0) { shutdown(fd,  SHUT_RDWR); close(fd);  fd  = -1; }
        if (rfd >= 0) { shutdown(rfd, SHUT_RDWR); close(rfd); rfd = -1; }
    };
    auto ring_connect = [&]() -> bool {             // clears every stage's KV (per-connection reset)
        ring_close();
        size_t c = connect_to.find(':');
        fd = tcp_connect(connect_to.substr(0, c), atoi(connect_to.substr(c + 1).c_str()));
        if (fd < 0) { fprintf(stderr, "server[pipe]: downstream connect failed\n"); return false; }
        rfd = tcp_listen_accept(return_listen);
        if (rfd < 0) { fprintf(stderr, "server[pipe]: return accept failed\n"); ring_close(); return false; }
        return true;
    };
    auto all_free = [&]() -> bool { for (auto & s : slots) if (s.phase != FREE) return false; return true; };

    // Chunk a job's prompt into wave slices (<= pf_wave; split any exactly-(k+1)-row wave into k,1).
    auto build_pf_waves = [&](PipeSlot & s) {
        s.pf_tok.clear(); s.pf_base.clear();
        const std::vector<int32_t> & p = s.job->prompt; const int total = (int) p.size();
        auto emit = [&](int b0, int n) { s.pf_tok.emplace_back(p.begin() + b0, p.begin() + b0 + n); s.pf_base.push_back(b0); };
        for (int base = 0; base < total; ) {
            int w = (total - base < pf_wave) ? (total - base) : pf_wave;
            if (w == mtp_k + 1) { emit(base, mtp_k); emit(base + mtp_k, 1); } else emit(base, w);
            base += w;
        }
        if (s.pf_tok.empty()) { s.pf_tok.emplace_back(p.begin(), p.end()); s.pf_base.push_back(0); }
    };
    // Emit m.out tokens into a job (honouring eog / max_tok). Returns true if the job is now finished.
    auto feed_out = [&](PipeSlot & s, const mtp_msg & m) -> bool {
        auto & job = *s.job; bool fin = false;
        std::lock_guard<std::mutex> lk(job.mtx);
        for (int t : m.out) {
            if (llama_vocab_is_eog(b.vocab, (llama_token) t)) { fin = true; break; }
            job.toks.push_back(t); s.produced++;
            if (s.produced >= s.max_tok) { fin = true; break; }
        }
        if (m.eog) fin = true;
        if (fin) job.done = true;
        job.cv.notify_all();
        return fin;
    };
    auto finish_slot = [&](PipeSlot & s) {           // caller holds M
        if (s.job) { std::lock_guard<std::mutex> lk(s.job->mtx); s.job->done = true; s.job->cv.notify_all(); }
        s.job.reset(); s.phase = FREE; s.pf_tok.clear(); s.pf_base.clear();
        s.pf_sent = s.pf_recv = 0; s.nx_issue.clear(); s.decode_ready = false; s.produced = 0;
        // s.seq is retired (its KV lingers until an idle reconnect recycles the seq id).
    };
    auto fail_all = [&]() {                          // caller holds M: connection died -> drop everything
        for (auto & s : slots) if (s.phase != FREE && s.job) {
            std::lock_guard<std::mutex> lk(s.job->mtx); s.job->failed = s.job->done = true; s.job->cv.notify_all();
        }
        for (auto & s : slots) { s.job.reset(); s.phase = FREE; s.inflight = 0; s.pf_tok.clear(); s.pf_base.clear();
                                 s.pf_sent = s.pf_recv = 0; s.nx_issue.clear(); s.decode_ready = false; s.produced = 0; }
        inflight_slot.clear();
    };

    if (!ring_connect()) { fprintf(stderr, "server[pipe]: initial ring connect FAILED\n"); return; }

    // ---- backward thread: recv one mtp_msg per outstanding wave, FIFO-match to slot, advance it ----
    std::thread bwd([&] {
        for (;;) {
            int cur_rfd;
            {   std::unique_lock<std::mutex> lk(M);
                CV.wait(lk, [&] { return stop || !inflight_slot.empty(); });
                if (stop && inflight_slot.empty()) return;
                cur_rfd = rfd;
            }
            mtp_msg m;
            if (!recv_mtp_msg(cur_rfd, m)) {           // return edge died: drop the whole connection
                std::lock_guard<std::mutex> lk(M);
                if (stop) return;
                fprintf(stderr, "server[pipe]: return edge dead -> failing %zu in-flight, reconnecting\n", inflight_slot.size());
                fail_all();
                if (!ring_connect()) { stop = true; CV.notify_all(); return; }
                CV.notify_all();
                continue;
            }
            std::unique_lock<std::mutex> lk(M);
            if (inflight_slot.empty()) continue;       // (post-reconnect) stale msg; ignore
            int si = inflight_slot.front(); inflight_slot.pop_front();
            PipeSlot & s = slots[si];
            if (s.inflight > 0) s.inflight--;
            if (s.phase == PREFILL) {
                s.pf_recv++;
                if (s.pf_recv >= s.pf_tok.size()) {    // FINAL prefill response == first generated token
                    bool fin = feed_out(s, m);
                    if (fin || m.issue.empty()) finish_slot(s);
                    else { s.nx_issue = m.issue; s.nx_base = m.p_base; s.decode_ready = true; s.phase = DECODE; }
                }
                // else: intermediate prefill response -> discard; fwd keeps sending the remaining waves.
            } else if (s.phase == DECODE) {
                bool fin = feed_out(s, m);
                if (fin || m.issue.empty()) finish_slot(s);
                else { s.nx_issue = m.issue; s.nx_base = m.p_base; s.decode_ready = true; }
            }
            CV.notify_all();
        }
    });

    // ---- forward thread: bind pending jobs to free slots, issue one wave per iteration ----
    std::thread fwd([&] {
        for (;;) {
            int    send_slot = -1, send_base = 0;
            std::vector<int32_t> send_tok;
            {   std::unique_lock<std::mutex> lk(M);
                CV.wait(lk, [&] {
                    if (stop) return true;
                    for (auto & s : slots) if (s.phase != FREE && s.decode_ready && s.inflight == 0) return true;
                    for (auto & s : slots) if (s.phase == PREFILL && s.pf_sent < s.pf_tok.size() && s.inflight < PF_DEPTH) return true;
                    if (!pending.empty()) for (auto & s : slots) if (s.phase == FREE)
                        return (next_seq < n_seq_max) || (inflight_slot.empty() && all_free());
                    return false;
                });
                if (stop) break;

                // 1) Bind as many pending jobs to free slots as we can.
                while (!pending.empty()) {
                    int fs = -1; for (int i = 0; i < n_slots; ++i) if (slots[i].phase == FREE) { fs = i; break; }
                    if (fs < 0) break;
                    if (next_seq >= n_seq_max) {                 // recycle seq ids: only safe when fully idle
                        if (inflight_slot.empty() && all_free()) { if (!ring_connect()) { stop = true; break; } next_seq = 0; }
                        else break;                              // wait for the ring to drain
                    }
                    PipeSlot & s = slots[fs];
                    s.job = pending.front(); pending.pop_front();
                    s.seq = next_seq++; s.phase = PREFILL; s.pf_sent = s.pf_recv = 0; s.inflight = 0;
                    s.produced = 0; s.max_tok = s.job->max_tok; s.decode_ready = false;
                    build_pf_waves(s);
                    { std::lock_guard<std::mutex> jk(s.job->mtx); s.job->started = true; s.job->cv.notify_all(); }
                }
                if (stop) break;

                // 2) Pick ONE wave to issue (decode-ready first, then a prefill chunk).
                for (int i = 0; i < n_slots && send_slot < 0; ++i) {
                    PipeSlot & s = slots[i];
                    if (s.phase != FREE && s.decode_ready && s.inflight == 0) {
                        send_slot = i; send_tok = s.nx_issue; send_base = s.nx_base;
                        s.decode_ready = false; s.inflight++; inflight_slot.push_back(i);
                    }
                }
                for (int i = 0; i < n_slots && send_slot < 0; ++i) {
                    PipeSlot & s = slots[i];
                    if (s.phase == PREFILL && s.pf_sent < s.pf_tok.size() && s.inflight < PF_DEPTH) {
                        send_slot = i; send_tok = s.pf_tok[s.pf_sent]; send_base = s.pf_base[s.pf_sent];
                        s.pf_sent++; s.inflight++; inflight_slot.push_back(i);
                    }
                }
                if (send_slot < 0) continue;
            }
            // 3) Head decode (stage 0) + send downstream -- OUTSIDE the lock so bwd/other work overlaps.
            const int seq = slots[send_slot].seq;
            std::vector<int32_t> sq(send_tok.size(), seq), ps(send_tok.size());
            for (size_t i = 0; i < send_tok.size(); ++i) ps[i] = send_base + (int) i;
            hidden_blob h;
            bool ok = run_tokens(b, send_tok, sq, ps, h) && send_hidden(fd, h);
            if (!ok) {
                std::lock_guard<std::mutex> lk(M);
                if (stop) break;
                fprintf(stderr, "server[pipe]: forward send failed -> failing in-flight, reconnecting\n");
                fail_all();
                if (!ring_connect()) { stop = true; CV.notify_all(); break; }
                CV.notify_all();
            } else {
                CV.notify_all();   // wave is now in flight -> wake bwd to receive its return (and re-arm fwd)
            }
        }
        stop = true; CV.notify_all();
    });

    // ---- HTTP front end: each request submits a job and drains its token queue ----
    auto submit = [&](std::vector<int32_t> ptoks, int max_tok) -> std::shared_ptr<PipeJob> {
        auto job = std::make_shared<PipeJob>();
        job->prompt = std::move(ptoks); job->max_tok = max_tok;
        { std::lock_guard<std::mutex> lk(M); if (stop) { job->done = job->failed = true; return job; } pending.push_back(job); }
        CV.notify_all();
        return job;
    };
    auto next_tok = [](PipeJob & j, int32_t & out) -> bool {   // false once done AND drained
        std::unique_lock<std::mutex> lk(j.mtx);
        j.cv.wait(lk, [&] { return !j.toks.empty() || j.done; });
        if (!j.toks.empty()) { out = j.toks.front(); j.toks.pop_front(); return true; }
        return false;
    };

    httplib::Server srv;
    auto handle = [&](const json & req, bool chat, httplib::Response & res) {
        std::vector<llama_token> ptoks;
        if (chat) {
            if (!req.contains("messages")) { res.status = 400; res.set_content("{\"error\":\"missing messages\"}", "application/json"); return; }
            ptoks = sv_tokenize(b, sv_apply_chat_template(b, req["messages"]), true);
        } else ptoks = sv_tokenize(b, req.value("prompt", std::string("")), true);
        int max_tok = req.value("max_tokens", default_max_tokens);
        if (max_tok <= 0) max_tok = default_max_tokens;
        if ((int) ptoks.size() >= per_seq_ctx) { res.status = 400; res.set_content("{\"error\":\"prompt exceeds per-seq ctx\"}", "application/json"); return; }
        max_tok = std::min(max_tok, per_seq_ctx - (int) ptoks.size() - 1);
        std::vector<int32_t> pt(ptoks.begin(), ptoks.end());
        bool stream = req.value("stream", false);

        if (stream) {
            auto job = submit(std::move(pt), max_tok);
            res.set_chunked_content_provider("text/event-stream",
                [&, job, chat](size_t, httplib::DataSink & sink) -> bool {
                    bool first = true; int32_t tok;
                    while (next_tok(*job, tok)) {
                        std::string piece = sv_detok(b, tok);
                        json d = {{"id","chatcmpl-stage"},{"object", chat?"chat.completion.chunk":"text_completion"},{"model", model_id},
                                  {"choices", json::array({ chat
                                     ? json{{"index",0},{"delta", first? json{{"role","assistant"},{"content",piece}} : json{{"content",piece}}},{"finish_reason",nullptr}}
                                     : json{{"index",0},{"text",piece},{"finish_reason",nullptr}} })}};
                        std::string sdat = "data: " + d.dump() + "\n\n"; first = false;
                        if (!sink.write(sdat.data(), sdat.size())) return false;
                    }
                    std::string done = "data: [DONE]\n\n"; sink.write(done.data(), done.size()); sink.done();
                    return true;
                });
        } else {
            auto job = submit(std::move(pt), max_tok);
            std::string text; int n = 0; int32_t tok;
            while (next_tok(*job, tok)) { text += sv_detok(b, tok); n++; }
            json resp = {{"id","chatcmpl-stage"},{"object", chat?"chat.completion":"text_completion"},{"model", model_id},
                {"choices", json::array({ chat
                   ? json{{"index",0},{"message", json{{"role","assistant"},{"content",text}}},{"finish_reason","stop"}}
                   : json{{"index",0},{"text",text},{"finish_reason","stop"}} })},
                {"usage", json{{"prompt_tokens",(int)ptoks.size()},{"completion_tokens",n},{"total_tokens",(int)ptoks.size()+n}}}};
            res.set_content(resp.dump(), "application/json");
        }
    };
    srv.Post("/v1/chat/completions", [&](const httplib::Request & rq, httplib::Response & res) {
        try { handle(json::parse(rq.body), true, res); }
        catch (const std::exception & e) { res.status = 400; res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json"); }
    });
    srv.Post("/v1/completions", [&](const httplib::Request & rq, httplib::Response & res) {
        try { handle(json::parse(rq.body), false, res); }
        catch (const std::exception & e) { res.status = 400; res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json"); }
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object","list"},{"data", json::array({ json{{"id",model_id},{"object","model"},{"owned_by","xautonomics"}} })}};
        res.set_content(j.dump(), "application/json");
    });
    fprintf(stderr, "server[pipe]: listening.\n");
    srv.listen("0.0.0.0", http_port);

    { std::lock_guard<std::mutex> lk(M); stop = true; } CV.notify_all();
    ring_close();
    fwd.join(); bwd.join();
}

static void run_server(model_bundle & b, const std::string & connect_to, int return_listen,
                       int http_port, int default_max_tokens, const std::string & model_id) {
    using json = nlohmann::json;
    const int n_ctx = (int) llama_n_ctx(b.ctx);
    fprintf(stderr, "server: ring=%s return-listen=:%d n_ctx=%d. OpenAI API on http://0.0.0.0:%d\n",
            connect_to.c_str(), return_listen, n_ctx, http_port);

    // --- persistent ring state (guarded by gen_mutex) ---
    int fd = -1, rfd = -1;               // ring downstream + tail's direct return edge (both plain TCP in the ik v1)
    std::vector<llama_token> cached;     // tokens currently resident in the ring KV (positions 0..size-1)
    auto gen_mutex = std::make_shared<std::mutex>();

    auto ring_close = [&]() {
        if (fd  >= 0) { shutdown(fd,  SHUT_RDWR); close(fd);  fd  = -1; }
        if (rfd >= 0) { shutdown(rfd, SHUT_RDWR); close(rfd); rfd = -1; }
        cached.clear();
    };
    // (Re)connect the ring. The new connection makes every stage clear its KV (its accept-loop reset),
    // and the tail reconnects its return edge, which we accept here.
    auto ring_connect = [&]() -> bool {
        ring_close();
        size_t c = connect_to.find(':');
        fd = tcp_connect(connect_to.substr(0, c), atoi(connect_to.substr(c + 1).c_str()));
        if (fd < 0) { fprintf(stderr, "server: downstream connect %s failed\n", connect_to.c_str()); return false; }
        rfd = tcp_listen_accept(return_listen);   // tail's return edge (arrives via the connection cascade)
        if (rfd < 0) { fprintf(stderr, "server: tail return accept failed\n"); ring_close(); return false; }
        return true;
    };

    // Return-edge recv with self-healing accept: the conn accepted at ring_connect time can be a
    // GHOST (a stale tail retry loop from a previous ring cycle wins the return-port race, then EOFs
    // at first read while the real tail is still forming the cascade). The forward wave is unaffected
    // (already in flight, KV intact), so on EOF we drop the conn and re-accept instead of tearing
    // the ring down. Bounded: each ghost costs one fast accept+recv cycle.
    auto recv_mtp_ret = [&](int & rfd_ref, mtp_msg & m) -> bool {
        for (int tries = 0; tries < 60; ++tries) {
            if (rfd_ref >= 0 && recv_mtp_msg(rfd_ref, m)) return true;
            if (rfd_ref >= 0) { shutdown(rfd_ref, SHUT_RDWR); close(rfd_ref); rfd_ref = -1; }
            fprintf(stderr, "server: return edge dead -- re-accepting :%d (try %d)\n", return_listen, tries + 1);
            rfd_ref = tcp_listen_accept(return_listen);
            if (rfd_ref < 0) return false;
        }
        return false;
    };

    // Generate one request. Prefix-matches `prompt` against the cached KV, prefills only the suffix
    // (append) when possible, reconnects (full reset) only on divergence / first use. Calls on_tok
    // per emitted token; on_tok==false stops. Returns #tokens generated.
    auto gen = [&](const std::vector<llama_token> & prompt, int max_tok, const std::function<bool(int)> & on_tok) -> int {
        // Prefix-cache (append only the new tokens) is ON by default (validated: multi-turn recall
        // correct via the tail re-prime fix); set STAGE_NO_PREFIX_CACHE to force full-prefill each request.
        static bool allow_cache = getenv("STAGE_NO_PREFIX_CACHE") == nullptr;
        static int mtp_k = []{ const char * e = getenv("STAGE_MTP_NDRAFT"); int k = e ? atoi(e) : 1; return k < 1 ? 1 : (k > 7 ? 7 : k); }();
        size_t L = 0; if (allow_cache) while (L < cached.size() && L < prompt.size() && cached[L] == prompt[L]) ++L;
        // reset (full prefill) only if there is no connection or the new prompt diverges from the cache.
        // An append whose delta happens to be k+1 rows no longer forces a reset: the tail re-primes on
        // ANY non-(k+1)-row wave, and we split any would-be k+1-row prefill wave into (k,1) below, so
        // no prefill/append wave is ever mistaken for a verify wave. Appends therefore always stay fast.
        bool reset = (fd < 0) || (L < cached.size());
        if (reset) { if (!ring_connect()) return 0; L = 0; }
        // Re-seed the last accepted token on the append path. A partial-accept turn-end leaves the tail's
        // KV at position L-1 holding a REJECTED draft: the tail dictated a next-verify wave that would
        // have re-decoded L-1 with the accepted token, but generation stopped and that wave never went
        // out. Re-decoding position L-1 here repairs it before the new suffix attends to it. (The tail
        // re-prime then rebuilds its draft state around this wave.) Also covers the identical-prompt case.
        size_t start = L;
        if (!reset && L > 0) start = L - 1;
        if (start >= prompt.size()) start = prompt.size() ? prompt.size() - 1 : 0;
        const size_t total = prompt.size();
        const size_t prefill_n = (total > start) ? total - start : 0;
        fprintf(stderr, "server: req prompt=%zu cached=%zu prefix=%zu prefill=%zu %s\n",
                prompt.size(), cached.size(), L, prefill_n, reset ? "(reset)" : "(append)");
        if (prefill_n == 0) { ring_close(); return 0; }
        // PREFILL IN WAVES: send <= STAGE_PREFILL_WAVE tokens per wave instead of one giant wave.
        // A single multi-thousand-token wave overruns the pipeline (a node errors on the huge
        // n_rows*n_embd*4 blob / long chunked decode -> the whole ring disconnects). Small waves
        // stay in the proven append regime; KV accumulates across waves. Intermediate waves' mtp
        // replies are mid-prompt samples (discarded); only the FINAL wave's m generates.
        static int pf_wave = []{ const char * e = getenv("STAGE_PREFILL_WAVE"); int w = e ? atoi(e) : 256; return w < 1 ? 256 : w; }();
        // Build wave SIZES, chunking to <= pf_wave. HARD GUARANTEE: no wave is exactly mtp_k+1 rows --
        // a primed tail re-primes on any non-(k+1)-row wave, so a k+1-row prefill/append wave would be
        // misread as a verify wave. Any wave that would be k+1 rows is split into (k, 1); for k>=1 both
        // k and 1 are strictly < k+1, so neither sub-wave can be mistaken either.
        std::vector<size_t> wsz;
        for (size_t rem = prefill_n; rem > 0; ) {
            size_t w = rem < (size_t) pf_wave ? rem : (size_t) pf_wave;
            if (w == (size_t) (mtp_k + 1)) { wsz.push_back((size_t) mtp_k); wsz.push_back(1); }
            else                           { wsz.push_back(w); }
            rem -= w;
        }
        mtp_msg m;
        size_t woff = start;
        for (size_t wi = 0; wi < wsz.size(); ++wi) {
            const size_t wend = woff + wsz[wi];
            std::vector<int32_t> tk, sq, ps;
            for (size_t p = woff; p < wend; ++p) { tk.push_back(prompt[p]); sq.push_back(0); ps.push_back((int) p); }
            hidden_blob h;
            if (!run_tokens(b, tk, sq, ps, h)) { fprintf(stderr, "server: prefill run_tokens FAILED (wave %zu/%zu rows=%zu)\n", wi + 1, wsz.size(), tk.size()); ring_close(); return 0; }
            errno = 0;
            if (!send_hidden(fd, h))           { fprintf(stderr, "server: prefill send_hidden FAILED errno=%d (%s)\n", errno, strerror(errno)); ring_close(); return 0; }
            errno = 0;
            if (!recv_mtp_ret(rfd, m))         { fprintf(stderr, "server: prefill recv_mtp_msg FAILED errno=%d (%s)\n", errno, strerror(errno)); ring_close(); return 0; }
            woff = wend;
        }
        std::vector<llama_token> generated;
        int n = 0; bool stop = false;
        for (;;) {
            for (int t : m.out) {
                if (llama_vocab_is_eog(b.vocab, (llama_token) t)) { stop = true; break; }
                generated.push_back((llama_token) t);
                if (!on_tok(t)) { stop = true; break; }
                if (++n >= max_tok) { stop = true; break; }
            }
            if (stop || m.eog || m.issue.empty()) break;
            std::vector<int32_t> dt = m.issue, ds(m.issue.size(), 0), dp(m.issue.size());
            for (size_t i = 0; i < m.issue.size(); ++i) dp[i] = m.p_base + (int) i;
            hidden_blob hh;
            if (!run_tokens(b, dt, ds, dp, hh)) { fprintf(stderr, "server: decode run_tokens FAILED\n"); ring_close(); return n; }
            errno = 0;
            if (!send_hidden(fd, hh))           { fprintf(stderr, "server: decode send_hidden FAILED errno=%d (%s)\n", errno, strerror(errno)); ring_close(); return n; }
            errno = 0;
            if (!recv_mtp_ret(rfd, m))          { fprintf(stderr, "server: decode recv_mtp_msg FAILED errno=%d (%s)\n", errno, strerror(errno)); ring_close(); return n; }
        }
        cached = prompt; cached.insert(cached.end(), generated.begin(), generated.end());   // KV now holds prompt+generated
        return n;
    };

    httplib::Server srv;
    auto handle = [&, gen_mutex](const json & req, bool chat, httplib::Response & res) {
        std::vector<llama_token> ptoks;
        if (chat) {
            if (!req.contains("messages")) { res.status = 400; res.set_content("{\"error\":\"missing messages\"}", "application/json"); return; }
            ptoks = sv_tokenize(b, sv_apply_chat_template(b, req["messages"]), true);
        } else {
            ptoks = sv_tokenize(b, req.value("prompt", std::string("")), true);
        }
        int max_tok = req.value("max_tokens", default_max_tokens);
        if (max_tok <= 0) max_tok = default_max_tokens;
        if ((int) ptoks.size() >= n_ctx) { res.status = 400; res.set_content("{\"error\":\"prompt exceeds n_ctx\"}", "application/json"); return; }
        max_tok = std::min(max_tok, n_ctx - (int) ptoks.size() - 1);   // keep prompt+gen within n_ctx
        bool stream = req.value("stream", false);

        if (stream) {
            res.set_chunked_content_provider("text/event-stream",
                [&, gen_mutex, ptoks, max_tok, chat, model_id](size_t, httplib::DataSink & sink) -> bool {
                    std::lock_guard<std::mutex> lk(*gen_mutex);
                    bool first = true;
                    gen(ptoks, max_tok, [&](int tok) -> bool {
                        std::string piece = sv_detok(b, tok);
                        json d = {{"id","chatcmpl-stage"},{"object", chat?"chat.completion.chunk":"text_completion"},{"model", model_id},
                                  {"choices", json::array({ chat
                                     ? json{{"index",0},{"delta", first? json{{"role","assistant"},{"content",piece}} : json{{"content",piece}}},{"finish_reason",nullptr}}
                                     : json{{"index",0},{"text",piece},{"finish_reason",nullptr}} })}};
                        std::string s = "data: " + d.dump() + "\n\n"; first = false;
                        return sink.write(s.data(), s.size());
                    });
                    std::string done = "data: [DONE]\n\n"; sink.write(done.data(), done.size()); sink.done();
                    return true;
                });
        } else {
            std::lock_guard<std::mutex> lk(*gen_mutex);
            std::string text;
            int n = gen(ptoks, max_tok, [&](int tok) -> bool { text += sv_detok(b, tok); return true; });
            json resp = {{"id","chatcmpl-stage"},{"object", chat?"chat.completion":"text_completion"},{"model", model_id},
                {"choices", json::array({ chat
                   ? json{{"index",0},{"message", json{{"role","assistant"},{"content",text}}},{"finish_reason","stop"}}
                   : json{{"index",0},{"text",text},{"finish_reason","stop"}} })},
                {"usage", json{{"prompt_tokens",(int)ptoks.size()},{"completion_tokens",n},{"total_tokens",(int)ptoks.size()+n}}}};
            res.set_content(resp.dump(), "application/json");
        }
    };

    srv.Post("/v1/chat/completions", [&](const httplib::Request & rq, httplib::Response & res) {
        try { handle(json::parse(rq.body), true, res); }
        catch (const std::exception & e) { res.status = 400; res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json"); }
    });
    srv.Post("/v1/completions", [&](const httplib::Request & rq, httplib::Response & res) {
        try { handle(json::parse(rq.body), false, res); }
        catch (const std::exception & e) { res.status = 400; res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json"); }
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object","list"},{"data", json::array({ json{{"id",model_id},{"object","model"},{"owned_by","xautonomics"}} })}};
        res.set_content(j.dump(), "application/json");
    });
    fprintf(stderr, "server: listening.\n");
    errno = 0;
    bool _lok = srv.listen("0.0.0.0", http_port);
    fprintf(stderr, "server: srv.listen RETURNED %d errno=%d (%s) is_valid=%d\n",
            (int) _lok, errno, strerror(errno), (int) srv.is_valid());
    ring_close();
}
