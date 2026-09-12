// CPU self-consistency of the quantized matmul path.
//
// For every quantized type the CPU backend can quantize, compare
//   mul_mat(quantized W, X)            -- the CPU's own kernel for that type
// against
//   mul_mat(f32 W', X), W' = cast(quantized W, f32)   -- the same weights through the type's dequantizer
// at n = 1, 7 and 32 columns. The two differ only by the activation quantization the kernel
// applies, which sits at NMSE ~1e-5; anything above the threshold means the kernel and the
// dequantizer disagree about what the bits mean. test-backend-ops cannot see this: it uses
// the CPU as its reference, so a wrong CPU kernel shows up there as every other backend
// failing that type.
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

static double nmse(const float * a, const float * b, size_t n) {
    double e = 0, s = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        e += d*d;
        s += (double) b[i]*b[i];
    }
    return s > 0 ? e / s : (e > 0 ? INFINITY : 0.0);
}

static bool run_type(ggml_type type, int64_t m, int64_t k, double tol, bool verbose) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> wf(m*k);
    for (auto & x : wf) x = dist(rng);

    const size_t row_size = ggml_row_size(type, k);
    std::vector<uint8_t> wq(m*row_size);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, m, k, nullptr, nullptr);

    bool ok = true;
    for (int n : {1, 7, 32}) {
        std::vector<float> xf(n*k);
        for (auto & x : xf) x = dist(rng);

        ggml_init_params ip = { (size_t) 256*1024*1024 + 2*m*k*sizeof(float), nullptr, false };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        memcpy(a->data, wq.data(), wq.size());
        memcpy(b->data, xf.data(), xf.size()*sizeof(float));

        ggml_tensor * c    = ggml_mul_mat(ctx, a, b);
        ggml_tensor * af   = ggml_cast(ctx, a, GGML_TYPE_F32);
        ggml_tensor * cref = ggml_mul_mat(ctx, af, b);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
        ggml_build_forward_expand(gf, cref);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        const double err = nmse((const float *) c->data, (const float *) cref->data, m*n);
        const bool pass = err < tol;
        ok = ok && pass;
        if (verbose || !pass) {
            printf("  %-8s m=%-5lld n=%-3d k=%-6lld NMSE(kernel vs dequant) = %.3e %s\n",
                    ggml_type_name(type), (long long) m, n, (long long) k, err, pass ? "OK" : "FAIL");
        }
        ggml_free(ctx);
    }
    return ok;
}

int main(int argc, char ** argv) {
    const bool verbose = argc > 1 && std::string(argv[1]) == "-v";
    const double tol = 1e-3;
    int n_fail = 0, n_run = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        const ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        if (!ggml_is_quantized(type) || traits.from_float == nullptr || traits.to_float == nullptr ||
            ggml_quantize_requires_imatrix(type)) {
            continue;
        }
        const int64_t blck = ggml_blck_size(type);
        if (blck <= 0) {
            continue;
        }
        // not self-contained as a (quantize, dequantize) pair: the row-interleaved *_r4 / *_r8
        // layouts are produced by repacking, and the BitNet types keep their row scale elsewhere
        const std::string name = ggml_type_name(type);
        if (name.find("_r") != std::string::npos || name.find("_bn") != std::string::npos) {
            continue;
        }
        // two model-like shapes: a wide-output projection and a square block
        for (auto shape : { std::pair<int64_t, int64_t>{320, 10240}, std::pair<int64_t, int64_t>{2560, 2560} }) {
            const int64_t m = shape.first, k = shape.second;
            if (k % blck != 0) {
                continue;
            }
            ++n_run;
            if (!run_type(type, m, k, tol, verbose)) {
                ++n_fail;
            }
        }
    }
    printf("%d/%d type/shape combinations consistent\n", n_run - n_fail, n_run);
    return n_fail == 0 ? 0 : 1;
}
