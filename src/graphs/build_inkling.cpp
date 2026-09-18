#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <vector>

// Inkling (TML, PRIVATE arch) — D2 CPU-parity graph.
//
// Ported from the internal reference (project 10, branch agent/noah/inkling-p1,
// src/models/inkling.cpp @ 681 lines). That tree is the mainline llama.cpp lineage
// (llm_graph_context + llama_memory_hybrid_iswa); this fork is the ik lineage
// (llm_build_context + the single llama_kv_cache). The reconciles that matter:
//
//   reference (mainline)                     this fork (ik)
//   ------------------------------------     --------------------------------------
//   llama_memory_hybrid_iswa_context         kv_self: k_l/v_l for attn, s_l for conv
//   mctx_recr->get_r_l(il)                   kv_self.s_l[il]
//   ggml_ssm_conv(ctx, sx, kernel)           ggml_ssm_conv(ctx, s, x, c, sq, saved)
//     3-arg, state concat'd by the caller      6-arg, fused: returns conv output AND
//                                              the new state in one tensor (see the
//                                              openPangu precedent in this tree)
//   build_attn(..., kq_b, ...)               llm_build_kv() has NO kq_b parameter, so
//                                              the relative-position bias needs the
//                                              bespoke KQ path below
//   separate base/swa caches + banded FA     one cache + SWA masks; the banded
//                                              flash-attn op is D3, NOT D2
//
// D2 scope is the CPU correctness path only. Per docs/inkling-d1-arch-plumbing.md:31-41
// there is deliberately no banded-FA op here — correctness uses the ordinary SWA mask
// path — and the hybrid/recurrent upstream memory classes are not ported. The conv
// state rides this tree's native kv_self.s_l, allocated in llama.cpp alongside the
// openPangu hybrid path.
//
// Per-layer shape: h += attn_sconv(attn(attn_norm(h))); h += mlp_sconv(mlp(mlp_norm(h)))
// Four short-conv streams share one packed state cell: [k | v | attn | mlp].

// One short-conv site. sconv(x) = x + causal_depthwise_conv1d(x), rolling state = last
// K-1 inputs. `state_all` is this layer's kv_self.s_l slot table; `off` is this stream's
// float offset into the packed cell. Mirrors openpangu_causal_conv() in this tree, but
// with a general d_conv (openPangu hard-codes kernel 3 => 2 taps) and four sites.
static ggml_tensor * inkling_sconv(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * x,            // {w, n_tokens}
        ggml_tensor  * kernel,       // {K, w}
        ggml_tensor  * state_all,    // {n_embd_r, n_state_slots}
        int64_t        off,          // float offset of this stream in the packed cell
        int64_t        d_conv,       // K - 1
        ggml_tensor  * seq_ids,      // I32 {1, n_tokens}
        bool           reset_state) {
    const int64_t w = x->ne[0];
    const int64_t T = x->ne[1];

    GGML_ASSERT(state_all != nullptr);
    GGML_ASSERT(state_all->type == GGML_TYPE_F32);
    GGML_ASSERT(state_all->ne[0] >= off + d_conv*w);
    GGML_ASSERT(seq_ids != nullptr && seq_ids->type == GGML_TYPE_I32);

    const size_t esz = ggml_element_size(state_all);

    // ggml_ssm_conv wants an f32 kernel; the checkpoint stores these f16/quantised.
    ggml_tensor * wc = ggml_reshape_2d(ctx, ggml_cast(ctx, kernel, GGML_TYPE_F32), d_conv + 1, w);

    // this stream's slice of the packed cell, slot 0 (single-sequence decode)
    ggml_tensor * state_flat = ggml_view_2d(ctx, state_all, d_conv*w, 1,
            state_all->nb[1], off*esz);

    // pos-0 graphs are discarded from reuse, so a baked reset never runs at pos > 0
    ggml_tensor * state_in = reset_state ? ggml_scale(ctx, state_flat, 0.0f) : state_flat;
    ggml_tensor * states   = ggml_reshape_3d(ctx, state_in, d_conv, w, 1);

    // fused: conv_raw carries the conv output followed by the updated taps
    ggml_tensor * conv_raw = ggml_ssm_conv(ctx, states, x, wc, seq_ids, nullptr);

    ggml_tensor * conv = ggml_view_2d(ctx, conv_raw, w, T,
            w*ggml_element_size(conv_raw), 0);

    // built-in residual, no activation (reference: y = x3 + conv_out)
    ggml_tensor * out = ggml_add(ctx, x, conv);

    // write the trailing d_conv columns back into the cache slot
    ggml_tensor * new_states = ggml_view_2d(ctx, conv_raw, d_conv, w,
            (d_conv + 1)*ggml_element_size(conv_raw),
            (1 + w*T)*ggml_element_size(conv_raw));
    ggml_tensor * new_states_cont = ggml_cont(ctx, new_states);
    ggml_tensor * new_state_flat  = ggml_reshape_2d(ctx, new_states_cont, d_conv*w, 1);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state_flat, state_flat));

    return out;
}

ggml_cgraph * llm_build_context::build_inkling() {
    // D2 is the CPU correctness path. Inkling's content-relative bias is a per-head
    // additive term on kq, and ggml_flash_attn_ext never materialises kq -- the reference
    // handles that with GGML_OP_FLASH_ATTN_EXT_BANDED, which is D3 work and not in this
    // tree yet. Until it lands, the arch must run with flash attention OFF.
    GGML_ASSERT(!cparams.flash_attn &&
            "INKLING requires -fa 0 until the banded flash-attn op (D3) lands: the "
            "content-relative bias is per-head and cannot be applied on the FA path");

    ggml_cgraph * gf = new_graph_custom();

    const int64_t d_rel    = hparams.inkling_d_rel;
    const int64_t d_conv   = hparams.n_shortconv_l_cache - 1;
    const int64_t n_embd_r = hparams.n_embd_r();
    const int64_t kw_max   = hparams.n_embd_k_gqa_max();
    const int64_t vw_max   = hparams.n_embd_v_gqa_max();

    // packed conv-state stream offsets within one cell: [k | v | attn | mlp]
    const int64_t off_k    = 0;
    const int64_t off_v    = d_conv*kw_max;
    const int64_t off_attn = d_conv*(kw_max + vw_max);
    const int64_t off_mlp  = d_conv*(kw_max + vw_max + n_embd);

    GGML_ASSERT(n_embd_r >= off_mlp + d_conv*n_embd);

    const int64_t n_vocab = model.vocab.n_tokens();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // mtmd embd rows arrive pre-normalised; embed_norm applies to text token lookups only
    if (batch.token) {
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(inpL, "inkling_embd_norm", -1);
    } else {
        cb(inpL, "inkling_mm_embd", -1);
    }
    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : nullptr;

    // sequence ids for ggml_ssm_conv, shared by all four conv sites
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);
    ggml_tensor * seq_ids = lctx.inp_s_seq_qnext;

    // log-N attention scaling, global (non-SWA) layers only: tau = 1 + alpha*log(max(pos+1)/n_floor, 1)
    ggml_tensor * tau = nullptr;
    if (hparams.inkling_log_n_floor > 0) {
        lctx.inp_inkling_tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        cb(lctx.inp_inkling_tau, "inp_inkling_tau", -1);
        ggml_set_input(lctx.inp_inkling_tau);
        tau = lctx.inp_inkling_tau;
    }

    // flattened relative-position indices, one per (kv, token) pair; index E selects the
    // zero pad column for out-of-band / empty cells
    const int32_t n_kv_cur = n_kv;
    lctx.inp_inkling_rel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_cur, n_tokens);
    cb(lctx.inp_inkling_rel_idx, "inp_inkling_rel_idx", -1);
    ggml_set_input(lctx.inp_inkling_rel_idx);
    ggml_tensor * rel_idx = lctx.inp_inkling_rel_idx;

    ggml_tensor * rel_idx_swa = nullptr;
    if (hparams.n_swa > 0) {
        lctx.inp_inkling_rel_idx_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_cur, n_tokens);
        cb(lctx.inp_inkling_rel_idx_swa, "inp_inkling_rel_idx_swa", -1);
        ggml_set_input(lctx.inp_inkling_rel_idx_swa);
        rel_idx_swa = lctx.inp_inkling_rel_idx_swa;
    }

    // padded vocab rows get -inf so samplers never emit a padded id
    ggml_tensor * vocab_mask = nullptr;
    if (!cparams.embeddings && hparams.inkling_unpadded_n_vocab > 0 &&
        (int64_t) hparams.inkling_unpadded_n_vocab < n_vocab) {
        lctx.inp_inkling_vocab_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_vocab);
        cb(lctx.inp_inkling_vocab_mask, "inp_inkling_vocab_mask", -1);
        ggml_set_input(lctx.inp_inkling_vocab_mask);
        vocab_mask = lctx.inp_inkling_vocab_mask;
    }

    // shared experts go through mul_mat_id: 2D views into a repacked/quantised 3D bank
    // are invalid, so the constant 0..n_shexp-1 index tensor is an input
    const int64_t n_shexp = hparams.n_expert_shared;
    ggml_tensor * shexp_idx = nullptr;
    if (n_shexp > 0 && (uint32_t) n_layer > hparams.n_layer_dense_lead) {
        lctx.inp_inkling_shexp_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_shexp, n_tokens);
        cb(lctx.inp_inkling_shexp_idx, "inp_inkling_shexp_idx", -1);
        ggml_set_input(lctx.inp_inkling_shexp_idx);
        shexp_idx = lctx.inp_inkling_shexp_idx;
    }

    // states are zeroed at cache allocation; a pos-0 batch resets them explicitly
    const bool reset_state = batch.pos && batch.pos[0] == 0;

    cur = inpL;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * state_all = kv_self.s_l[il];
        GGML_ASSERT(state_all != nullptr);

        const bool    is_swa     = hparams.is_swa(il);
        const int64_t n_head_kv  = hparams.n_head_kv(il);
        const int64_t head_dim   = hparams.n_embd_head_k(il);
        const int64_t rel_extent = is_swa ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;

        ggml_tensor * inpSA = cur;

        // ---- attention sublayer: h += attn_sconv(attn(attn_norm(h))) ----
        ggml_tensor * attn_in = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(attn_in, "inkling_attn_norm", il);

        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq, attn_in);
        ggml_tensor * k = llm_build_lora_mm(lctx, ctx0, layer.wk, attn_in);
        ggml_tensor * v = llm_build_lora_mm(lctx, ctx0, layer.wv, attn_in);
        ggml_tensor * r = llm_build_lora_mm(lctx, ctx0, layer.wr, attn_in);
        cb(q, "inkling_attn_q", il);
        cb(k, "inkling_attn_k", il);
        cb(v, "inkling_attn_v", il);
        cb(r, "inkling_attn_r", il);

        // k/v short convs on the flat projections, BEFORE the head reshape
        k = inkling_sconv(ctx0, gf, k, layer.shortconv_k, state_all, off_k, d_conv, seq_ids, reset_state);
        v = inkling_sconv(ctx0, gf, v, layer.shortconv_v, state_all, off_v, d_conv, seq_ids, reset_state);
        cb(k, "inkling_attn_k_sconv", il);
        cb(v, "inkling_attn_v_sconv", il);

        q = ggml_reshape_3d(ctx0, q, head_dim, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_tokens);

        q = llm_build_norm(ctx0, q, hparams, layer.attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
        k = llm_build_norm(ctx0, k, hparams, layer.attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(q, "inkling_attn_q_norm", il);
        cb(k, "inkling_attn_k_norm", il);

        // log-N tau on global layers only, after q_norm
        if (tau && !is_swa) {
            q = ggml_mul(ctx0, q, tau);
        }

        // ---- relative-position bias ----
        // proj stored [E, d_rel]; transpose so ggml_mul_mat contracts over d_rel
        ggml_tensor * r2   = ggml_reshape_2d(ctx0, r, d_rel, n_head*n_tokens);
        ggml_tensor * proj = ggml_cont(ctx0, ggml_transpose(ctx0, layer.attn_rel_proj));

        ggml_tensor * rel = ggml_mul_mat(ctx0, proj, r2);
        ggml_mul_mat_set_prec(rel, GGML_PREC_F32);
        rel = ggml_reshape_3d(ctx0, rel, rel_extent, n_head, n_tokens);

        if (tau && !is_swa) {
            rel = ggml_mul(ctx0, rel, tau);
        }
        cb(rel, "inkling_rel_logits", il);

        // zero column at index E is gathered by out-of-band / empty-cell indices
        rel = ggml_pad(ctx0, rel, 1, 0, 0, 0);                              // {E+1, n_head, n_tokens}
        rel = ggml_cont(ctx0, ggml_permute(ctx0, rel, 1, 0, 2, 3));         // {n_head, E+1, n_tokens}
        rel = ggml_reshape_2d(ctx0, rel, n_head, (rel_extent + 1)*n_tokens);

        ggml_tensor * idx = is_swa ? rel_idx_swa : rel_idx;
        GGML_ASSERT(idx != nullptr);

        ggml_tensor * idx1 = ggml_reshape_1d(ctx0, idx, n_kv_cur*n_tokens);
        ggml_tensor * kq_b = ggml_get_rows(ctx0, rel, idx1);                // {n_head, n_kv*n_tokens}
        kq_b = ggml_reshape_3d(ctx0, kq_b, n_head, n_kv_cur, n_tokens);
        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));       // {n_kv, n_tokens, n_head}
        cb(kq_b, "inkling_kq_b", il);

        // ---- attention with the relative bias as a true per-head kq_b ----
        // The bias CANNOT ride kq_mask: ggml asserts mask->ne[2] == 1 (head broadcast)
        // while this bias is per-head {n_kv, n_tokens, n_head}. It is threaded to
        // llm_build_kqv as an optional kq_b and added to kq before the softmax.
        //
        // The reference divides q by head_dim (NOT sqrt(head_dim)) and passes kq_scale 1.0
        // so the bias stays unscaled -- keep that exactly, it is parity-visible.
        q = ggml_scale(ctx0, q, 1.0f/float(head_dim));

        ggml_tensor * kq_mask_cur = is_swa && KQ_mask_swa ? KQ_mask_swa : KQ_mask;
        const int n_swa_l = is_swa ? (int) hparams.n_swa : 0;

        ggml_tensor * attn_out = llm_build_kv(ctx0, lctx, kv_self, gf,
                layer.wo, NULL,
                k, v, q,
                kq_mask_cur,
                n_tokens, kv_head, n_kv_cur,
                1.0f,
                cb, il,
                nullptr, n_swa_l, -1,
                nullptr, nullptr, -1,
                kq_b);
        cb(attn_out, "inkling_attn_o", il);

        attn_out = inkling_sconv(ctx0, gf, attn_out, layer.shortconv_attn, state_all,
                off_attn, d_conv, seq_ids, reset_state);
        cb(attn_out, "inkling_attn_sconv", il);

        cur = ggml_add(ctx0, inpSA, attn_out);

        // ---- feed-forward sublayer: h += mlp_sconv(mlp(mlp_norm(h))) ----
        ggml_tensor * ffn_res = cur;

        ggml_tensor * ffn_in = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(ffn_in, "inkling_ffn_norm", il);

        ggml_tensor * ffn_out;

        if (il < (int) hparams.n_layer_dense_lead) {
            ffn_out = llm_build_ffn(ctx0, lctx, nullptr, ffn_in,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            ffn_out = ggml_mul(ctx0, ffn_out, layer.ffn_gscale);
            cb(ffn_out, "inkling_dense_ffn_out", il);
        } else {
            // Custom routing, not expressible via llm_build_moe_ffn: select by
            // top-k(sigmoid(logits) + bias), weight by softmax(logsigmoid(raw logits)).
            ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, ffn_in);
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
            cb(logits, "inkling_moe_logits", il);

            const size_t lsz = ggml_element_size(logits);

            ggml_tensor * routed = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, logits, n_expert, n_tokens, logits->nb[1], 0));
            ggml_tensor * shared_logits = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, logits, n_shexp, n_tokens, logits->nb[1], n_expert*lsz));

            // bias affects selection only, not the weights
            ggml_tensor * scores = ggml_sigmoid(ctx0, routed);
            scores = ggml_add(ctx0, scores, layer.ffn_exp_probs_b);
            cb(scores, "inkling_moe_scores", il);

            // ggml_top_k is argsort + a view, so the result is NOT contiguous. It is used
            // below as ggml_get_rows indices and as mul_mat_id expert ids, both of which
            // need a real buffer -- glm5next:176 and deepseek4:885 wrap it the same way.
            ggml_tensor * selected = ggml_cont(ctx0, ggml_top_k(ctx0, scores, n_expert_used));
            cb(selected, "inkling_moe_topk", il);

            ggml_tensor * routed3     = ggml_reshape_3d(ctx0, routed, 1, n_expert, n_tokens);
            ggml_tensor * topk_logits = ggml_get_rows(ctx0, routed3, selected);
            topk_logits = ggml_reshape_2d(ctx0, topk_logits, n_expert_used, n_tokens);

            ggml_tensor * all_logits = ggml_concat(ctx0, topk_logits, shared_logits, 0);

            // logsigmoid(x) = -softplus(-x); the softmax spans routed top-k AND shared
            // logits together, so shared gammas fall out of the same distribution
            ggml_tensor * w = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, all_logits)));
            w = ggml_soft_max(ctx0, w);
            w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
            // gate global scale applies to the MoE weights too, not just the dense FFN
            w = ggml_mul(ctx0, w, layer.ffn_gscale);
            cb(w, "inkling_moe_weights", il);

            const size_t wsz = ggml_element_size(w);

            ggml_tensor * weights = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, w, n_expert_used, n_tokens, w->nb[1], 0));
            weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);

            ggml_tensor * xr = ggml_reshape_3d(ctx0, ffn_in, n_embd, 1, n_tokens);

            // Routed experts are applied directly rather than via llm_build_moe_ffn: that
            // helper derives its own weights from [n_expert, n_tokens] probs and cannot
            // express inkling's softmax(logsigmoid(.)) taken jointly over routed+shared.
            ggml_tensor * moe_out = nullptr;
            {
                ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_gate_exps, xr, selected);
                ggml_tensor * up   = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_up_exps,   xr, selected);
                ggml_tensor * h    = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);

                ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_down_exps, h, selected);
                experts = ggml_mul(ctx0, experts, weights);

                for (int64_t i = 0; i < n_expert_used; ++i) {
                    ggml_tensor * e = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                            experts->nb[2], i*experts->nb[1]);
                    moe_out = moe_out ? ggml_add(ctx0, moe_out, e) : e;
                }
            }

            // shared experts, weighted by the trailing n_shexp gammas of the same softmax
            if (n_shexp > 0) {
                GGML_ASSERT(shexp_idx != nullptr);

                ggml_tensor * gs = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_gate_shexp, xr, shexp_idx);
                ggml_tensor * us = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_up_shexp,   xr, shexp_idx);
                ggml_tensor * hs = ggml_mul(ctx0, ggml_silu(ctx0, gs), us);

                // gammas must scale hs BEFORE the down-proj to match reference rounding
                ggml_tensor * gammas = ggml_cont(ctx0,
                        ggml_view_2d(ctx0, w, n_shexp, n_tokens, w->nb[1], n_expert_used*wsz));
                hs = ggml_mul(ctx0, hs, ggml_reshape_3d(ctx0, gammas, 1, n_shexp, n_tokens));

                ggml_tensor * ds = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_down_shexp, hs, shexp_idx);

                for (int64_t sx = 0; sx < n_shexp; ++sx) {
                    ggml_tensor * e = ggml_view_2d(ctx0, ds, n_embd, n_tokens, ds->nb[2], sx*ds->nb[1]);
                    moe_out = ggml_add(ctx0, moe_out, e);
                }
            }
            cb(moe_out, "inkling_moe_out", il);
            ffn_out = moe_out;
        }

        ffn_out = inkling_sconv(ctx0, gf, ffn_out, layer.shortconv_mlp, state_all,
                off_mlp, d_conv, seq_ids, reset_state);
        cb(ffn_out, "inkling_ffn_sconv", il);

        cur = ggml_add(ctx0, ffn_res, ffn_out);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
    }

    // conv states need every layer to see ALL tokens, so trim outputs only after the
    // full stack — never hoist this into the loop.
    if (n_tokens > 1) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    if (!cparams.embeddings) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        ggml_mul_mat_set_prec(cur,
                model.output->type == GGML_TYPE_F32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);

        if (vocab_mask) {
            cur = ggml_add(ctx0, cur, vocab_mask);
        }
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
