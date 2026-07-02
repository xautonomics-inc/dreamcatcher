// OpenAI-compatible server: ik-llama-stage-runner-as-head driving the multi-stage MTP ring.
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
        // reset (full prefill) if: no connection, prompt diverges from cache, or the append delta is
        // exactly 1+K rows (a tail verify wave is 1+K rows -> indistinguishable; the tail would misread it).
        bool reset = (fd < 0) || (L < cached.size()) || ((int) (prompt.size() - L) == mtp_k + 1);
        if (reset) { if (!ring_connect()) return 0; L = 0; }
        size_t start = L;
        if (start >= prompt.size()) start = prompt.size() ? prompt.size() - 1 : 0;   // identical prompt: re-seed last token
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
        // Build wave end-offsets; ensure NO wave is exactly mtp_k+1 rows (a primed tail would misread
        // it as a verify wave). Only a trailing remainder can be small -> borrow a row from the prev wave.
        std::vector<size_t> ends;
        for (size_t off = start; off < total; ) { size_t e = (off + (size_t) pf_wave < total) ? off + (size_t) pf_wave : total; ends.push_back(e); off = e; }
        if (ends.size() >= 2 && (int) (total - ends[ends.size() - 2]) == mtp_k + 1) ends[ends.size() - 2] -= 1;
        mtp_msg m;
        size_t woff = start;
        for (size_t wi = 0; wi < ends.size(); ++wi) {
            const size_t wend = ends[wi];
            std::vector<int32_t> tk, sq, ps;
            for (size_t p = woff; p < wend; ++p) { tk.push_back(prompt[p]); sq.push_back(0); ps.push_back((int) p); }
            hidden_blob h;
            if (!run_tokens(b, tk, sq, ps, h)) { fprintf(stderr, "server: prefill run_tokens FAILED (wave %zu/%zu rows=%zu)\n", wi + 1, ends.size(), tk.size()); ring_close(); return 0; }
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
