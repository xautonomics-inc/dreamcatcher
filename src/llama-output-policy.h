#pragma once

// Output-buffer policy: the single source of truth for whether a context has a
// logits buffer at all.
//
// Two places must agree, or llama_decode dereferences a buffer that was never
// allocated:
//   * llama_output_reserve() sizes lctx.logits (0 rows => lctx.logits == nullptr)
//   * llama_decode() decides whether to extract logits out of the graph
// When they disagree the symptom is
//   GGML_ASSERT(lctx.logits != nullptr) failed
// from the logits-extraction block of llama_decode.
//
// Rule: a context produces logits unless it runs in embeddings mode. An MTP
// context is the exception - its NextN/MTP head consumes logits even while the
// hidden state is exported alongside them - so it keeps the logits buffer.
//
// Regression guard (see tests/test-output-policy.cpp): "the model carries
// NextN/MTP weights" (hparams.nextn_predict_layers > 0) is NOT the same
// question as "this context decodes with an MTP head" (cparams.mtp, via
// llama_context_has_mtp_outputs). A hidden-state emit stage - STAGE_EMIT=hidden
// in llama-stage-runner, used for library-vs-monolith A/B and for head stages -
// sets embeddings=true with mtp=false on a NextN model. Its graph stops at
// result_norm and never builds output_norm+lm_head, so there is no logits tensor
// to extract and no logits buffer is reserved. Keying this predicate off
// nextn_predict_layers would ask for logits that neither the graph nor the
// buffer can supply.
static inline bool llama_output_has_logits(bool embeddings, bool has_mtp_outputs) {
    return !embeddings || has_mtp_outputs;
}
