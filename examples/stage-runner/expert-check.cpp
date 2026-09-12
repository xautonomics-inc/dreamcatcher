// expert-check: byte-exactness + mechanism-overhead harness for expert-tensor
// disaggregation (P1).
//
// Loads a model, evaluates a fixed prompt, then greedy-decodes N tokens with a
// pure argmax (no sampler chain). For every step it prints the token id and an
// FNV-1a 64 hash over the full logits vector, and optionally appends the raw
// logits bytes to the file named by EXPERT_CHECK_LOGITS_OUT. Two runs are
// equivalent iff their outputs are byte-identical:
//
//   baseline : llama-expert-check -m M.gguf -t 24 ...
//   split    : LLAMA_EXPERTS_REMOTE=127.0.0.1:9666 llama-expert-check ... (same args)
//   proof    : diff the stdouts; cmp the logits dumps
//
// If the build repacks quantized tensors on load, the baseline must disable it
// (-rtr 0 / --no-repack, whichever this build exposes): the expert-server
// computes on the raw tensor bytes, so the baseline has to as well.
//
// Timing: prefill and decode are reported separately; decode explicitly
// excludes the prompt pass so the per-token delta between the two modes
// isolates serialization + framing + loopback/network cost.

#include "common.h"
#include "llama.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t fnv1a64(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt    = "The capital of France is";
    params.n_predict = 20;
    params.warmup    = false;   // deterministic harness; no warm-up decode

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result li = llama_init_from_gpt_params(params);
    llama_model   * model = li.model;
    llama_context * ctx   = li.context;
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "expert-check: failed to load model\n");
        return 1;
    }

    const int n_vocab = llama_n_vocab(model);

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, /*add_special*/ true, /*parse_special*/ true);
    if (toks.empty()) {
        fprintf(stderr, "expert-check: empty prompt\n");
        return 1;
    }
    printf("PROMPT n_tokens=%zu\n", toks.size());

    FILE * lf = nullptr;
    const char * lo = getenv("EXPERT_CHECK_LOGITS_OUT");
    if (lo && lo[0]) {
        lf = fopen(lo, "wb");
        if (!lf) { fprintf(stderr, "expert-check: cannot open %s\n", lo); return 1; }
    }

    // ---- prefill -----------------------------------------------------------
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i == toks.size() - 1);
    }

    double t0 = now_ms();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "expert-check: prefill decode failed\n");
        return 1;
    }
    double t_prefill = now_ms() - t0;

    // ---- greedy decode -----------------------------------------------------
    int       n_gen  = 0;
    double    t_gen  = 0;
    llama_pos pos    = (llama_pos) toks.size();
    std::string text;

    for (int step = 0; step < params.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);

        // pure argmax, lowest id wins ties - deterministic given identical bytes
        int   best   = 0;
        float best_v = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }

        printf("TOK step=%d id=%d logits_fnv=0x%016" PRIx64 "\n",
               step, best, fnv1a64(logits, (size_t) n_vocab * sizeof(float)));
        if (lf) {
            fwrite(logits, sizeof(float), n_vocab, lf);
        }
        text += common_token_to_piece(ctx, best);

        if (llama_token_is_eog(model, best)) {
            printf("EOG at step %d\n", step);
            break;
        }

        common_batch_clear(batch);
        common_batch_add(batch, best, pos++, { 0 }, true);

        double g0 = now_ms();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "expert-check: decode failed at step %d\n", step);
            return 1;
        }
        t_gen += now_ms() - g0;
        n_gen++;
    }

    printf("TEXT %s\n", text.c_str());
    printf("SUMMARY n_prefill=%zu prefill_ms=%.1f prefill_tok_s=%.2f n_gen=%d gen_ms=%.1f decode_tok_s=%.3f\n",
           toks.size(), t_prefill, 1000.0 * toks.size() / t_prefill,
           n_gen, t_gen, n_gen > 0 ? 1000.0 * n_gen / t_gen : 0.0);

    if (lf) fclose(lf);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
