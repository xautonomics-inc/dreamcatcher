#include "../src/llama-output-policy.h"

#include <cstdio>

// The logits buffer is allocated by llama_output_reserve() and filled by
// llama_decode(). Both ask llama_output_has_logits(); this pins the answer so
// the two can never drift apart again.
//
// The historical bug was that llama_decode() additionally keyed off
// hparams.nextn_predict_layers, so on a NextN model in embeddings mode it tried
// to extract logits that llama_output_reserve() had not allocated:
//   GGML_ASSERT(lctx.logits != nullptr) failed
// That is the STAGE_EMIT=hidden path (embeddings=true, mtp=false) on
// GLM-5.3-Flash / glm5next. nextn_predict_layers is a property of the *model*;
// whether logits are produced is a property of the *context*.

static int failures = 0;

static void check(bool embeddings, bool has_mtp, bool expected, const char * what) {
    const bool got = llama_output_has_logits(embeddings, has_mtp);
    if (got != expected) {
        std::printf("FAIL %s: embeddings=%d has_mtp=%d -> %d, expected %d\n",
                what, (int) embeddings, (int) has_mtp, (int) got, (int) expected);
        ++failures;
    }
}

int main() {
    // generation: logits are the point
    check(false, false, true,  "generation");
    // generation on an MTP context: still logits (the MTP head consumes them)
    check(false, true,  true,  "generation+mtp");
    // embeddings-only: the graph stops before lm_head, no logits buffer exists.
    // This is the regression case - it must hold for NextN models too, which is
    // why the predicate cannot look at nextn_predict_layers.
    check(true,  false, false, "embeddings (STAGE_EMIT=hidden)");
    // MTP context exporting hidden state: logits are still required
    check(true,  true,  true,  "embeddings+mtp");

    // The predicate must depend on nothing else: it is a total function of two
    // booleans, so enumerate it and confirm it equals the documented rule.
    for (int e = 0; e < 2; ++e) {
        for (int m = 0; m < 2; ++m) {
            const bool expected = !((bool) e) || ((bool) m);
            check((bool) e, (bool) m, expected, "truth table");
        }
    }

    if (failures) {
        std::printf("test-output-policy: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-output-policy: OK\n");
    return 0;
}
