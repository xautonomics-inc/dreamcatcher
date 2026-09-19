// CPU self-check for GGML_OP_FLASH_ATTN_EXT_BANDED (D3, Inkling).
//
// Runs the banded kernel on random inputs and compares against a plain double-precision
// re-implementation of its contract:
//   dist  = q_pos - kv_pos            (position inputs)  |  q_row + (n_kv - n_q) - k_col (default)
//   skip  if window > 0 and dist outside [0, window); skip if mask == -INF; skip empty cells
//   score = (q.k)*scale + (0 <= dist < E ? rel[dist, head, q_row, batch] : 0) + mask
// and cross-checks that the three ways of expressing the SWA cutoff (mask only, kernel
// window only, both) and the two distance conventions (positions vs. columns) agree
// bit-for-bit when they describe the same geometry.

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t g_rng = 1u;
static void  seed(uint32_t s) { g_rng = s; }
static float frand() {
    g_rng = g_rng*1664525u + 1013904223u;
    return ((g_rng >> 8) & 0xffff)/65536.0f*2.0f - 1.0f;
}

struct Case {
    std::string name;
    int64_t Dk, Dv, n_q, n_kv, n_head, n_head_kv, n_batch, E;
    ggml_type kv_type;      // F16 or F32
    ggml_type rel_type;     // F32 / F16 / BF16
    int64_t   rel_batch;    // 1 or n_batch
    bool use_pos;
    bool permute_cells;     // cell index != position (positional mode only)
    int64_t kv_head;        // first cell of the query block (positional mode)
    bool use_mask;
    int mask_window;        // SWA cutoff carried by the mask (0 = causal only)
    int op_window;          // SWA cutoff carried by the op
    float scale;
};

// rounds through the storage type the kernel will actually see
static float round_kv(float x, ggml_type t) {
    return t == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)) : x;
}
static float round_rel(float x, ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        default:             return x;
    }
}
static void store_rel(void * dst, ggml_type t, float x) {
    switch (t) {
        case GGML_TYPE_F16:  *(ggml_fp16_t *) dst = ggml_fp32_to_fp16(x); break;
        case GGML_TYPE_BF16: *(ggml_bf16_t *) dst = ggml_fp32_to_bf16(x); break;
        default:             *(float *) dst = x; break;
    }
}

static bool run_case(const Case & c, std::vector<float> & out_copy) {
    seed(0xC0FFEE);

    ggml_init_params ip = { 512ull*1024*1024, NULL, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t Dk = c.Dk, Dv = c.Dv, NQ = c.n_q, NK = c.n_kv, NH = c.n_head, NHK = c.n_head_kv, NB = c.n_batch, E = c.E;
    const size_t kv_sz = ggml_type_size(c.kv_type);

    // q is built in the model layout {Dk, n_head, n_q, n_batch} and permuted like the graph does
    ggml_tensor * q0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dk, NH, NQ, NB);
    std::vector<float> qh(Dk*NH*NQ*NB);                       // reference copy (rounded as the kernel sees it)
    for (int64_t i = 0; i < (int64_t) qh.size(); ++i) {
        const float x = frand();
        ((float *) q0->data)[i] = x;
        // q is converted to K's vec_dot type: F16 for an F16 K, F32 for F32
        qh[i] = round_kv(x, c.kv_type);
    }
    ggml_tensor * q = ggml_permute(ctx, q0, 0, 2, 1, 3);     // {Dk, n_q, n_head, n_batch}

    ggml_tensor * k = ggml_new_tensor_4d(ctx, c.kv_type, Dk, NK, NHK, NB);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, c.kv_type, Dv, NK, NHK, NB);
    std::vector<float> kh(Dk*NK*NHK*NB), vh(Dv*NK*NHK*NB);
    for (int64_t i = 0; i < (int64_t) kh.size(); ++i) {
        const float x = frand(); kh[i] = round_kv(x, c.kv_type);
        if (c.kv_type == GGML_TYPE_F16) ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(x); else ((float *) k->data)[i] = x;
    }
    for (int64_t i = 0; i < (int64_t) vh.size(); ++i) {
        const float x = frand(); vh[i] = round_kv(x, c.kv_type);
        if (c.kv_type == GGML_TYPE_F16) ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(x); else ((float *) v->data)[i] = x;
    }
    GGML_UNUSED(kv_sz);

    // band: {E, n_head, n_q, rel_batch}, values of a size that matters against q.k*scale
    ggml_tensor * rel = ggml_new_tensor_4d(ctx, c.rel_type, E, NH, NQ, c.rel_batch);
    std::vector<float> relh(E*NH*NQ*c.rel_batch);
    for (int64_t i = 0; i < (int64_t) relh.size(); ++i) {
        const float x = 2.0f*frand();
        store_rel((char *) rel->data + i*ggml_type_size(c.rel_type), c.rel_type, x);
        relh[i] = round_rel(x, c.rel_type);
    }

    // positions
    std::vector<int32_t> qpos(NQ), kpos(NK);
    if (c.use_pos) {
        // cells [0, kv_head + n_q) hold positions; permute_cells scrambles which cell holds which
        const int64_t n_live = c.kv_head + NQ;
        GGML_ASSERT(n_live <= NK);
        std::vector<int32_t> perm(n_live);
        for (int64_t i = 0; i < n_live; ++i) perm[i] = (int32_t) i;
        if (c.permute_cells) {
            for (int64_t i = n_live - 1; i > 0; --i) {
                const int64_t j = (int64_t) ((g_rng = g_rng*1664525u + 1013904223u) >> 8) % (i + 1);
                std::swap(perm[i], perm[j]);
            }
        }
        for (int64_t j = 0; j < NK; ++j) kpos[j] = j < n_live ? perm[j] : -1;
        for (int64_t i = 0; i < NQ; ++i) qpos[i] = (int32_t) (c.kv_head + i);
    } else {
        for (int64_t j = 0; j < NK; ++j) kpos[j] = (int32_t) j;
        for (int64_t i = 0; i < NQ; ++i) qpos[i] = (int32_t) (i + (NK - NQ));
    }

    ggml_tensor * t_qpos = nullptr, * t_kpos = nullptr;
    if (c.use_pos) {
        t_qpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ);
        t_kpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NK);
        memcpy(t_qpos->data, qpos.data(), NQ*sizeof(int32_t));
        memcpy(t_kpos->data, kpos.data(), NK*sizeof(int32_t));
    }

    // mask {n_kv, PAD(n_q)} f16: empty / causal / optional SWA, on the same positions
    ggml_tensor * mask = nullptr;
    std::vector<float> maskh(NK*NQ, 0.0f);
    if (c.use_mask) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NK, GGML_PAD(NQ, GGML_KQ_MASK_PAD));
        ggml_fp16_t * md = (ggml_fp16_t *) mask->data;
        for (int64_t i = 0; i < mask->ne[1]; ++i) {
            for (int64_t j = 0; j < NK; ++j) {
                float m = -INFINITY;
                if (i < NQ) {
                    const int32_t kp = kpos[j], qp = qpos[i];
                    if (kp >= 0 && kp <= qp && (c.mask_window == 0 || qp - kp < c.mask_window)) m = 0.0f;
                    maskh[i*NK + j] = m;
                }
                md[i*NK + j] = ggml_fp32_to_fp16(m);
            }
        }
    }

    ggml_tensor * out = ggml_flash_attn_ext_banded(ctx, q, k, v, mask, rel, c.scale, E);
    ggml_flash_attn_ext_banded_set_pos(out, t_qpos, t_kpos);
    ggml_flash_attn_ext_banded_set_window(out, c.op_window);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_cplan plan = ggml_graph_plan(gf, 4);
    std::vector<uint8_t> work(plan.work_size + 256);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    // reference
    const int64_t rk = NH/NHK;
    double max_err = 0.0; int64_t n_empty_rows = 0, n_bad = 0;
    std::vector<double> acc(Dv);
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t hk = h/rk;
            for (int64_t iq = 0; iq < NQ; ++iq) {
                const float * qrow = &qh[((b*NQ + iq)*NH + h)*Dk];
                const int64_t qp = c.use_pos ? qpos[iq] : iq + (NK - NQ);
                std::vector<double> sc(NK, -INFINITY);
                double M = -INFINITY;
                for (int64_t ic = 0; ic < NK; ++ic) {
                    int64_t dist;
                    if (c.use_pos) { if (kpos[ic] < 0) continue; dist = qp - kpos[ic]; }
                    else           { dist = qp - ic; }
                    if (c.op_window > 0 && (dist < 0 || dist >= c.op_window)) continue;
                    const float mv = c.use_mask ? maskh[iq*NK + ic] : 0.0f;
                    if (mv == -INFINITY) continue;
                    const float * krow = &kh[((b*NHK + hk)*NK + ic)*Dk];
                    double s = 0.0;
                    for (int64_t d = 0; d < Dk; ++d) s += (double) qrow[d]*krow[d];
                    s *= c.scale;
                    if (dist >= 0 && dist < E) s += relh[((b % c.rel_batch)*NQ + iq)*NH*E + h*E + dist];
                    s += mv;
                    sc[ic] = s; if (s > M) M = s;
                }
                std::fill(acc.begin(), acc.end(), 0.0);
                double S = 0.0;
                if (M != -INFINITY) {
                    for (int64_t ic = 0; ic < NK; ++ic) {
                        if (sc[ic] == -INFINITY) continue;
                        const double p = exp(sc[ic] - M); S += p;
                        const float * vrow = &vh[((b*NHK + hk)*NK + ic)*Dv];
                        for (int64_t d = 0; d < Dv; ++d) acc[d] += p*vrow[d];
                    }
                } else {
                    n_empty_rows++;
                }
                const float * o = (const float *) out->data + ((b*NQ + iq)*NH + h)*Dv;
                for (int64_t d = 0; d < Dv; ++d) {
                    const double ref = S > 0 ? acc[d]/S : 0.0;
                    const double err = fabs(ref - o[d]);
                    if (!(err <= 2e-4) || std::isnan(o[d])) n_bad++;
                    if (err > max_err) max_err = err;
                }
            }
        }
    }
    out_copy.assign((const float *) out->data, (const float *) out->data + ggml_nelements(out));
    printf("  %-44s max|err| = %.3e  empty rows = %lld  %s\n", c.name.c_str(), max_err, (long long) n_empty_rows, n_bad ? "FAIL" : "ok");
    ggml_free(ctx);
    return n_bad == 0;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

int main() {
    printf("test-flash-attn-banded: GGML_OP_FLASH_ATTN_EXT_BANDED CPU kernel vs double reference\n");
    bool ok = true;
    std::vector<float> o_col, o_col_as_pos, o_m, o_m_mask_only, o_m_win_only, o_perm, o_bf16, o_dec;

    //                  name                                    Dk  Dv  nq  nkv nh nhk nb  E   kv_type        rel_type        relb  pos    perm   kvhead mask  mwin owin  scale
    ok &= run_case({"column, f32 band, no mask, no window",     32, 32, 4,  16, 8, 2,  1,  6,  GGML_TYPE_F16, GGML_TYPE_F32,  1,    false, false, 0,     false, 0,  0,   1.0f/32}, o_col);
    ok &= run_case({"positions == columns (must equal above)",   32, 32, 4,  16, 8, 2,  1,  6,  GGML_TYPE_F16, GGML_TYPE_F32,  1,    true,  false, 12,    false, 0,  0,   1.0f/32}, o_col_as_pos);
    ok &= run_case({"padded cache, f16 band>window, mask+win 8", 32, 32, 5,  64, 8, 2,  1,  12, GGML_TYPE_F16, GGML_TYPE_F16,  1,    true,  false, 20,    true,  8,  8,   1.0f/32}, o_m);
    ok &= run_case({"  same geometry, SWA via mask only",        32, 32, 5,  64, 8, 2,  1,  12, GGML_TYPE_F16, GGML_TYPE_F16,  1,    true,  false, 20,    true,  8,  0,   1.0f/32}, o_m_mask_only);
    ok &= run_case({"  same geometry, SWA via op window only",   32, 32, 5,  64, 8, 2,  1,  12, GGML_TYPE_F16, GGML_TYPE_F16,  1,    true,  false, 20,    true,  0,  8,   1.0f/32}, o_m_win_only);
    ok &= run_case({"permuted cells, f32 kv, batch 2, global",   32, 32, 7,  40, 8, 2,  2,  9,  GGML_TYPE_F32, GGML_TYPE_F32,  2,    true,  true,  25,    true,  0,  0,   1.0f/32}, o_perm);
    ok &= run_case({"permuted cells, bf16 band, Dv != Dk, win 6",16, 24, 6,  48, 4, 4,  1,  6,  GGML_TYPE_F16, GGML_TYPE_BF16, 1,    true,  true,  30,    true,  6,  6,   1.0f/16}, o_bf16);
    ok &= run_case({"decode: n_q 1, tail of a 256-cell cache",   32, 32, 1,  256,8, 2,  1,  48, GGML_TYPE_F16, GGML_TYPE_F16,  1,    true,  false, 200,   true,  32, 32,  1.0f/32}, o_dec);

    const bool eq1 = same_bits(o_col, o_col_as_pos);
    const bool eq2 = same_bits(o_m, o_m_mask_only);
    const bool eq3 = same_bits(o_m, o_m_win_only);
    printf("  positions-vs-columns bitwise equal: %s\n", eq1 ? "yes" : "NO");
    printf("  mask-only vs mask+window bitwise equal: %s\n", eq2 ? "yes" : "NO");
    printf("  window-only vs mask+window bitwise equal: %s\n", eq3 ? "yes" : "NO");
    ok &= eq1 && eq2 && eq3;

    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
