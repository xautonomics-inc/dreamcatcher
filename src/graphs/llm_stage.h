#pragma once
// Multi-stage hidden-state pipeline config (xAutonomics fork).
// A "stage" process runs only a contiguous layer window [il_start, il_end) of the
// model and either emits the inter-layer residual hidden state (emit_hidden) for the
// next stage, or — if it owns the final transformer layer — finishes to logits.
// Driven by env so any arch builder / role honors it. See build_deepseek2.cpp.
#include <cstdlib>
#include <cstring>

struct llama_stage_cfg {
    bool active      = false;
    int  il_start    = 0;
    int  il_end      = 0;   // exclusive
    bool emit_hidden = false;
};

// n_proc_layers = hparams.n_layer - hparams.nextn_predict_layers (excludes the trailing NextN block).
static inline llama_stage_cfg llama_stage_get_cfg(int n_proc_layers) {
    llama_stage_cfg c;
    c.il_start = 0;
    c.il_end   = n_proc_layers;
    const char * a = getenv("STAGE_ACTIVE");
    if (!a || atoi(a) == 0) return c;            // inactive -> full model default
    c.active = true;
    if (const char * s = getenv("STAGE_IL_START")) c.il_start = atoi(s);
    if (const char * e = getenv("STAGE_IL_END"))   c.il_end   = atoi(e);
    if (c.il_start < 0) c.il_start = 0;
    if (c.il_end < 0 || c.il_end > n_proc_layers) c.il_end = n_proc_layers;
    if (c.il_end < c.il_start) c.il_end = c.il_start;
    const char * em = getenv("STAGE_EMIT");
    c.emit_hidden = em && strcmp(em, "hidden") == 0;
    return c;
}

// True when this window owns the model's final transformer layer (i.e. it is a tail).
static inline bool llama_stage_consumes_last(const llama_stage_cfg & c, int n_proc_layers) {
    const int last = n_proc_layers - 1;
    return last >= c.il_start && last < c.il_end;
}
