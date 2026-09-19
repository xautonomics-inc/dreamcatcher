// expert-server: expert-tensor disaggregation, server side (P1).
//
// Serves the routed-expert FFN of a MoE model over a framed TCP wire: loads
// ONLY blk.N.ffn_{gate,up,down}_exps.weight for the configured layers (mmap,
// zero-copy - everything else in the GGUF, shared experts included, is never
// faulted in), then answers EXPERT_CALL frames with EXPERT_RET frames.
//
// The per-request compute mirrors llm_build_context::llm_build_moe_ffn()'s
// routed-expert tail op-for-op on the same CPU kernels, so a split run is
// bit-exact against the single-process baseline: the router, selection and
// weight normalization ran attention-side through the untouched code path, and
// f32 hidden/weights cross the wire unrounded.
//
// This tree fuses parts of that tail by default, so the server has to mirror
// whichever form the client built:
//
//   --fmoe 1 (default)          --fmoe 0
//     par = moe_up_gate(          up   = mul_mat_id(up_exps,   hidden, ids)
//             up_exps,            gate = mul_mat_id(gate_exps, hidden, ids)
//             gate_exps,          par  = fused_mul_unary(gate, up, SILU)
//             hidden, ids, SILU)
//
//   down = mul_mat_id(down_exps, par, ids)
//
//   --mmad 1 (default)          --mmad 0
//     out = mul_multi_add(        wexp = mul(down, weights)
//             down, weights)      out  = multi_add(view_2d(wexp, ...), k)
//
// The defaults match this tree's client defaults (fused_moe_up_gate and
// fused_mmad are both on); a client run with -no-fmoe / -no-mmad needs the
// matching switch here.
//
// Not every architecture builds its routed tail through llm_build_moe_ffn.
// Inkling (src/graphs/build_inkling.cpp) routes over a joint softmax across
// routed AND shared experts and applies the routed tail directly, unfused:
//
//   --moe-form inkling
//     gate = mul_mat_id(gate_exps, hidden, ids)
//     up   = mul_mat_id(up_exps,   hidden, ids)
//     h    = mul(silu(gate), up)
//     e    = mul(mul_mat_id(down_exps, h, ids), weights)
//     out  = e[0] + e[1] + ... + e[k-1]        (one add per expert, in order)
//
// --fmoe / --mmad do not apply to that form. --moe-form auto (the default)
// picks it when the GGUF's general.architecture is "inkling" and the
// llm_build_moe_ffn form otherwise, so a server started on an Inkling file
// mirrors the right tail without an extra switch. The shared experts are
// never loaded here in either form: the server finds tensors by the
// ffn_*_exps names only, and the client keeps ffn_*_shexp in-process.
//
// Usage:
//   llama-expert-server --role expert-server --model M.gguf \
//       --expert-layers 1-47 --listen 9666 [--threads 32] [--host 0.0.0.0]
//       [--moe-form auto|moe_ffn|inkling] [--fmoe 0|1] [--mmad 0|1]
//       [--swiglu-limit F] [--no-prewarm] [--device cpu|gpu] [--gpu-chunk N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "llama-experts-remote.h"   // wire constants (magic values, frame layout)

#include <cerrno>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <map>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Which client-side routed-expert tail this server mirrors (see the header).
enum moe_form {
    MOE_FORM_MOE_FFN,   // llm_build_moe_ffn: --fmoe / --mmad select the fused variants
    MOE_FORM_INKLING,   // build_inkling: separate mul_mat_id, mul(silu(gate), up), per-expert adds
};

struct expert_layer {
    ggml_tensor * gate_exps = nullptr; // [n_embd, n_ff_exp, n_expert]
    ggml_tensor * up_exps   = nullptr;
    ggml_tensor * down_exps = nullptr; // [n_ff_exp, n_embd, n_expert]
};

static bool srv_send_all(int fd, const void * b, size_t n) {
    const char * p = (const char *) b;
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        p += k; n -= (size_t) k;
    }
    return true;
}

static bool srv_recv_all(int fd, void * b, size_t n) {
    char * p = (char *) b;
    while (n > 0) {
        ssize_t k = recv(fd, p, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        p += k; n -= (size_t) k;
    }
    return true;
}

static bool parse_layers(const char * s, std::vector<bool> & cov) {
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        long a = strtol(p, &end, 10);
        long b = a;
        if (end && *end == '-') b = strtol(end + 1, &end, 10);
        if (a < 0 || b < a || b >= 4096) return false;
        if ((long) cov.size() <= b) cov.resize(b + 1, false);
        for (long i = a; i <= b; ++i) cov[i] = true;
        if (end == nullptr || *end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string layers_spec;
    std::string bind_host = "0.0.0.0";
    int         listen_port  = 0;
    int         n_threads    = 16;
    bool        prewarm      = true;
    bool        use_fmoe     = true;    // mirror this tree's fused_moe_up_gate default
    bool        use_mmad     = true;    // mirror this tree's fused_mmad default
    float       swiglu_limit = 0.0f;    // nonzero only on the few archs that carry limits
    int         gpu_chunk    = 0;       // --gpu-chunk: layers per GPU buffer (0 = one buffer for all)
    bool        use_gpu      = false;   // --device gpu: expert compute on the first non-CPU backend
    std::string moe_form_arg = "auto";  // --moe-form: auto (from general.architecture) | moe_ffn | inkling

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](const char * nm) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "expert-server: %s needs an argument\n", nm); exit(1); }
            return argv[++i];
        };
        if      (a == "--role")          { std::string r = nx("--role");
                                           if (r != "expert-server") { fprintf(stderr, "expert-server: unsupported role '%s'\n", r.c_str()); return 1; } }
        else if (a == "--model" || a == "-m") model_path  = nx("--model");
        else if (a == "--expert-layers")      layers_spec = nx("--expert-layers");
        else if (a == "--listen")             listen_port = atoi(nx("--listen"));
        else if (a == "--host")               bind_host   = nx("--host");
        else if (a == "--threads" || a == "-t") n_threads = atoi(nx("--threads"));
        else if (a == "--fmoe")               use_fmoe    = atoi(nx("--fmoe")) != 0;
        else if (a == "--mmad")               use_mmad    = atoi(nx("--mmad")) != 0;
        else if (a == "--swiglu-limit")       swiglu_limit = (float) atof(nx("--swiglu-limit"));
        else if (a == "--moe-form")           moe_form_arg = nx("--moe-form");
        else if (a == "--device")             { std::string d = nx("--device");
                                                if (d == "gpu") use_gpu = true;
                                                else if (d != "cpu") { fprintf(stderr, "expert-server: --device must be cpu or gpu\n"); return 1; } }
        else if (a == "--no-prewarm")         prewarm     = false;
        else if (a == "--gpu-chunk")          gpu_chunk   = atoi(nx("--gpu-chunk"));
        else { fprintf(stderr, "expert-server: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || listen_port <= 0 || layers_spec.empty()) {
        fprintf(stderr, "usage: llama-expert-server --role expert-server --model M.gguf "
                        "--expert-layers a-b[,c,...] --listen PORT [--threads N] [--host H] "
                        "[--moe-form auto|moe_ffn|inkling] [--fmoe 0|1] [--mmad 0|1] "
                        "[--swiglu-limit F] [--device cpu|gpu] [--gpu-chunk N] [--no-prewarm]\n");
        return 1;
    }
    if (moe_form_arg != "auto" && moe_form_arg != "moe_ffn" && moe_form_arg != "inkling") {
        fprintf(stderr, "expert-server: --moe-form must be auto, moe_ffn or inkling (got '%s')\n", moe_form_arg.c_str());
        return 1;
    }

    std::vector<bool> cov;
    if (!parse_layers(layers_spec.c_str(), cov)) {
        fprintf(stderr, "expert-server: bad --expert-layers '%s'\n", layers_spec.c_str());
        return 1;
    }

    // ---- open the GGUF(s): metadata only, then mmap ------------------------
    //
    // Models big enough to want expert disaggregation are usually SPLIT. Open
    // every shard and search them all for the covered layers' exps tensors.
    // mmap is virtual, so shards holding no covered layer cost address space
    // and nothing else - a server for the back half of the stack never faults a
    // page of the shards holding the front half.

    struct model_shard {
        std::string    path;
        gguf_context * gguf = nullptr;
        ggml_context * meta = nullptr;
        uint8_t *      base = nullptr;
        size_t         data_off = 0;
        int            fd = -1;   // kept open: GPU mode uploads by pread, not by faulting the mmap
    };
    std::vector<model_shard> shards;

    auto open_shard = [&](const std::string & path) -> bool {
        model_shard sh;
        sh.path = path;
        gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &sh.meta };
        sh.gguf = gguf_init_from_file(path.c_str(), gp);
        if (sh.gguf == nullptr) {
            fprintf(stderr, "expert-server: cannot read gguf %s\n", path.c_str());
            return false;
        }
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror("expert-server: open model"); return false; }
        struct stat st = {};
        fstat(fd, &st);
        sh.base = (uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (sh.base == MAP_FAILED) { perror("expert-server: mmap model"); close(fd); return false; }
        sh.fd = fd;   // CPU mode serves straight out of the mapping; GPU mode preads from this fd
        sh.data_off = gguf_get_data_offset(sh.gguf);
        shards.push_back(sh);
        return true;
    };

    if (!open_shard(model_path)) {
        return 1;
    }

    // "...-00001-of-00006.gguf" -> open the remaining shards. Follow split.count
    // when the first shard declares it, otherwise fall back to the filename.
    {
        int n_split = 0;
        const int kid = gguf_find_key(shards[0].gguf, "split.count");
        if (kid >= 0) {
            switch (gguf_get_kv_type(shards[0].gguf, kid)) {
                case GGUF_TYPE_UINT16: n_split = gguf_get_val_u16(shards[0].gguf, kid); break;
                case GGUF_TYPE_UINT32: n_split = (int) gguf_get_val_u32(shards[0].gguf, kid); break;
                case GGUF_TYPE_INT32:  n_split = gguf_get_val_i32(shards[0].gguf, kid); break;
                default: break;
            }
        }
        const size_t dash = model_path.rfind('-', model_path.rfind("-of-") == std::string::npos
                                                 ? std::string::npos : model_path.rfind("-of-") - 1);
        const size_t ofp  = model_path.rfind("-of-");
        if (n_split > 1 && ofp != std::string::npos && dash != std::string::npos && dash < ofp) {
            const std::string prefix = model_path.substr(0, dash + 1);           // "...-"
            const std::string suffix = model_path.substr(ofp);                    // "-of-00006.gguf"
            for (int i = 2; i <= n_split; ++i) {
                char idx[16];
                snprintf(idx, sizeof(idx), "%05d", i);
                if (!open_shard(prefix + idx + suffix)) {
                    return 1;
                }
            }
            fprintf(stderr, "expert-server: model is split across %d shards\n", n_split);
        }
    }

    // ---- which routed tail to mirror ---------------------------------------
    std::string arch;
    {
        const int aid = gguf_find_key(shards[0].gguf, "general.architecture");
        if (aid >= 0 && gguf_get_kv_type(shards[0].gguf, aid) == GGUF_TYPE_STRING) {
            arch = gguf_get_val_str(shards[0].gguf, aid);
        }
    }
    moe_form form = MOE_FORM_MOE_FFN;
    if (moe_form_arg == "inkling" || (moe_form_arg == "auto" && arch == "inkling")) {
        form = MOE_FORM_INKLING;
    }
    if (moe_form_arg == "auto") {
        fprintf(stderr, "expert-server: general.architecture '%s' -> --moe-form %s\n",
                arch.c_str(), form == MOE_FORM_INKLING ? "inkling" : "moe_ffn");
    }

    // where a tensor's bytes live in the file, so GPU mode can read them explicitly
    std::map<const ggml_tensor *, std::pair<int, size_t>> src_of;   // tensor -> (fd, absolute offset)

    auto find_tensor = [&](const char * fmt, int il) -> ggml_tensor * {
        char name[256];
        snprintf(name, sizeof(name), fmt, il);
        for (model_shard & sh : shards) {
            const int tid = gguf_find_tensor(sh.gguf, name);
            if (tid < 0) continue;
            ggml_tensor * t = ggml_get_tensor(sh.meta, name);
            if (t == nullptr) continue;
            const size_t off = sh.data_off + gguf_get_tensor_offset(sh.gguf, tid);
            t->data = sh.base + off;
            src_of[t] = { sh.fd, off };
            return t;
        }
        return nullptr;
    };

    std::vector<expert_layer> layers(cov.size());
    int     n_loaded = 0;
    size_t  bytes    = 0;
    int64_t n_embd = 0, n_ff_exp = 0, n_expert = 0;
    ggml_type exps_type = GGML_TYPE_F32;

    for (size_t il = 0; il < cov.size(); ++il) {
        if (!cov[il]) continue;
        expert_layer & L = layers[il];
        L.gate_exps = find_tensor("blk.%d.ffn_gate_exps.weight", (int) il);
        L.up_exps   = find_tensor("blk.%d.ffn_up_exps.weight",   (int) il);
        L.down_exps = find_tensor("blk.%d.ffn_down_exps.weight", (int) il);
        if (!L.gate_exps || !L.up_exps || !L.down_exps) {
            fprintf(stderr, "expert-server: layer %zu has no ffn_*_exps tensors (dense layer or bad --expert-layers?)\n", il);
            return 1;
        }
        n_embd    = L.gate_exps->ne[0];
        n_ff_exp  = L.gate_exps->ne[1];
        n_expert  = L.gate_exps->ne[2];
        exps_type = L.gate_exps->type;
        bytes += ggml_nbytes(L.gate_exps) + ggml_nbytes(L.up_exps) + ggml_nbytes(L.down_exps);
        n_loaded++;
    }
    if (n_loaded == 0) {
        fprintf(stderr, "expert-server: --expert-layers selected no layers\n");
        return 1;
    }
    if (form == MOE_FORM_INKLING) {
        // build_inkling never fuses: neither switch has a client-side counterpart
        use_fmoe = false;
        use_mmad = false;
    }
    if (use_fmoe && layers[0].up_exps && layers[0].gate_exps &&
            layers[0].up_exps->type != layers[0].gate_exps->type) {
        // the client falls back to the separate path in exactly this case
        fprintf(stderr, "expert-server: up/gate expert types differ - forcing --fmoe 0 to match the client\n");
        use_fmoe = false;
    }

    fprintf(stderr, "expert-server: %d layers, n_embd %" PRId64 ", n_ff_exp %" PRId64 ", n_expert %" PRId64
                    ", %.2f GiB expert weights (%s, %s, form %s, fmoe %d, mmad %d)\n",
            n_loaded, n_embd, n_ff_exp, n_expert, bytes / (1024.0*1024.0*1024.0),
            ggml_type_name(exps_type), prewarm ? "prewarming" : "cold mmap",
            form == MOE_FORM_INKLING ? "inkling" : "moe_ffn", (int) use_fmoe, (int) use_mmad);

    if (prewarm && !use_gpu) {
        // fault the expert pages in now rather than on the first request
        volatile uint64_t sink = 0;
        double t0 = now_ms();
        for (size_t il = 0; il < cov.size(); ++il) {
            if (!cov[il]) continue;
            for (ggml_tensor * t : { layers[il].gate_exps, layers[il].up_exps, layers[il].down_exps }) {
                const uint8_t * p = (const uint8_t *) t->data;
                const size_t n = ggml_nbytes(t);
                for (size_t o = 0; o < n; o += 4096) sink += p[o];
            }
        }
        (void) sink;
        fprintf(stderr, "expert-server: prewarm done in %.1f s\n", (now_ms() - t0) / 1000.0);
    }

    // ---- GPU mode: upload the expert tensors into backend buffers ----------
    // One buffer per --gpu-chunk layers (a single very large allocation is
    // asking for trouble). On a unified-memory host these land in GTT - the
    // upload is a memory copy, and per-request I/O crosses no PCIe bus.
    //
    // This tree predates the backend-device API, so the backend is picked out
    // of the registry by name instead: the first registered backend that is not
    // the CPU one.
    ggml_backend_t gbackend = nullptr;
    if (use_gpu) {
        for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
            const char * nm = ggml_backend_reg_get_name(i);
            if (nm == nullptr || strcmp(nm, "CPU") == 0) continue;
            gbackend = ggml_backend_reg_init_backend(i, nullptr);
            if (gbackend != nullptr) {
                fprintf(stderr, "expert-server: GPU backend: %s\n", ggml_backend_name(gbackend));
                break;
            }
        }
        if (gbackend == nullptr) {
            fprintf(stderr, "expert-server: --device gpu but no non-CPU backend is registered\n");
            return 1;
        }

        // Read the bytes explicitly rather than handing ggml_backend_tensor_set a
        // pointer into the mmap: uploading N GiB that way faults N GiB of page
        // cache, which at full size is self-defeating - the device buffer has
        // already taken the RAM the page cache would need. Sequential pread into
        // a small reused bounce buffer, telling the kernel to drop each chunk
        // behind us, costs one streaming pass and a fixed amount of anonymous
        // memory.
        const size_t UP_CHUNK = 64ull * 1024 * 1024;
        std::vector<uint8_t> bounce(UP_CHUNK);
        auto upload_tensor = [&](ggml_tensor * dst, const ggml_tensor * src) -> bool {
            const size_t nbytes = ggml_nbytes(src);
            auto it = src_of.find(src);
            if (it == src_of.end()) {
                // no recorded file location (should not happen) - fall back to the mapping
                ggml_backend_tensor_set(dst, src->data, 0, nbytes);
                return true;
            }
            const int    fd  = it->second.first;
            const size_t off = it->second.second;
            size_t done = 0;
            while (done < nbytes) {
                const size_t want = std::min(UP_CHUNK, nbytes - done);
                size_t got = 0;
                while (got < want) {
                    const ssize_t r = pread(fd, bounce.data() + got, want - got, (off_t) (off + done + got));
                    if (r <= 0) {
                        if (r < 0 && errno == EINTR) continue;
                        fprintf(stderr, "expert-server: pread failed at offset %zu (%s)\n",
                                off + done + got, strerror(errno));
                        return false;
                    }
                    got += (size_t) r;
                }
                ggml_backend_tensor_set(dst, bounce.data(), done, want);
                // drop what we just read; nothing needs it again
                posix_fadvise(fd, (off_t) (off + done), (off_t) want, POSIX_FADV_DONTNEED);
                done += want;
            }
            return true;
        };

        double t0 = now_ms();
        size_t uploaded = 0;
        std::vector<size_t> todo;
        for (size_t il = 0; il < cov.size(); ++il) {
            if (cov[il]) todo.push_back(il);
        }
        const size_t chunk = gpu_chunk == 0 ? todo.size() : (size_t) gpu_chunk;
        size_t n_bufs = 0;
        for (size_t base = 0; base < todo.size(); base += chunk) {
            const size_t n_here = std::min(chunk, todo.size() - base);
            ggml_init_params wip = { (3 * n_here + 8) * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
            ggml_context * wctx = ggml_init(wip);
            std::vector<ggml_tensor *> dg(n_here), du(n_here), dd(n_here);
            for (size_t j = 0; j < n_here; ++j) {
                expert_layer & L = layers[todo[base + j]];
                dg[j] = ggml_dup_tensor(wctx, L.gate_exps);
                du[j] = ggml_dup_tensor(wctx, L.up_exps);
                dd[j] = ggml_dup_tensor(wctx, L.down_exps);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, gbackend);
            if (buf == nullptr) {
                fprintf(stderr, "expert-server: GPU buffer alloc failed at layer %zu "
                                "(%.2f GiB uploaded in %zu buffer(s), chunk %zu)\n",
                        todo[base], uploaded / (1024.0*1024.0*1024.0), n_bufs, chunk);
                return 1;
            }
            n_bufs++;
            for (size_t j = 0; j < n_here; ++j) {
                expert_layer & L = layers[todo[base + j]];
                if (!upload_tensor(dg[j], L.gate_exps)) return 1;
                if (!upload_tensor(du[j], L.up_exps))   return 1;
                if (!upload_tensor(dd[j], L.down_exps)) return 1;
                uploaded += ggml_nbytes(dg[j]) + ggml_nbytes(du[j]) + ggml_nbytes(dd[j]);
                // serve from the device tensors from here on
                L.gate_exps = dg[j]; L.up_exps = du[j]; L.down_exps = dd[j];
            }
        }
        fprintf(stderr, "expert-server: uploaded %.2f GiB to GPU in %.1f s (%zu buffer(s), chunk %zu)\n",
                uploaded / (1024.0*1024.0*1024.0), (now_ms() - t0) / 1000.0, n_bufs, chunk);
    }

    // ---- listen ------------------------------------------------------------

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa = {};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((uint16_t) listen_port);
    sa.sin_addr.s_addr = inet_addr(bind_host.c_str());
    if (bind(ls, (struct sockaddr *) &sa, sizeof(sa)) != 0) { perror("expert-server: bind"); return 1; }
    if (listen(ls, 4) != 0) { perror("expert-server: listen"); return 1; }
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "expert-server: listening on %s:%d (%d threads)\n", bind_host.c_str(), listen_port, n_threads);

    for (;;) {
        int cs = accept(ls, nullptr, nullptr);
        if (cs < 0) { if (errno == EINTR) continue; perror("expert-server: accept"); return 1; }
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::thread([&, cs]() {
        int32_t hello[2];
        if (!srv_recv_all(cs, hello, sizeof(hello)) || hello[0] != LLAMA_EXPERTS_MAGIC_CAPS ||
                hello[1] != LLAMA_EXPERTS_VERSION) {
            fprintf(stderr, "expert-server: invalid CAPS request\n");
            close(cs);
            return;
        }
        std::vector<int32_t> served;
        for (size_t i = 0; i < cov.size(); ++i) if (cov[i]) served.push_back((int32_t) i);
        std::vector<int32_t> caps = { LLAMA_EXPERTS_MAGIC_CAPS, LLAMA_EXPERTS_VERSION,
                                      (int32_t) served.size(), (int32_t) n_embd };
        caps.insert(caps.end(), served.begin(), served.end());
        caps.push_back(use_gpu ? 1 : 0);
        caps.push_back(n_threads);
        if (!srv_send_all(cs, caps.data(), caps.size() * sizeof(int32_t))) {
            close(cs);
            return;
        }
        fprintf(stderr, "expert-server: client connected (served %zu layers)\n", served.size());

        // connection-local allocator: two clients must not share one
        ggml_gallocr_t conn_galloc = use_gpu
            ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(gbackend)) : nullptr;

        std::vector<uint8_t> workbuf;
        std::vector<uint8_t> outbuf;
        std::vector<uint8_t> payload;

        uint64_t n_calls = 0;
        double   t_recv = 0, t_comp = 0, t_send = 0;

        // Phase totals split by call shape. idle is the gap between finishing one
        // reply and the first byte of the next request header - i.e. everything
        // that happens outside this process (client-side graph work + wire).
        // Comparing idle here against the client's asm/send/wait/recv says which
        // side of the socket a missing interval went to.
        struct srv_phase { uint64_t n=0, rows=0; double idle=0, recv=0, comp=0, send=0; };
        srv_phase ph_dec, ph_pre;
        double t_prev_done = now_ms();

        for (;;) {
            int32_t hdr[5];
            if (!srv_recv_all(cs, hdr, sizeof(hdr))) break;
            double t0 = now_ms();
            const double idle_ms = t0 - t_prev_done;
            if (hdr[0] != LLAMA_EXPERTS_MAGIC_CALL) {
                fprintf(stderr, "expert-server: bad magic %08x, dropping client\n", hdr[0]);
                break;
            }
            const int layer  = hdr[1];
            const int n_tok  = hdr[2];
            const int n_topk = hdr[3];
            const int c_embd = hdr[4];

            int32_t status = 0;
            if (layer < 0 || layer >= (int) layers.size() || !cov[layer])    status = 1; // layer not served
            if (c_embd != n_embd)                                            status = 2;
            if (n_tok <= 0 || n_tok > 65536 || n_topk <= 0 || n_topk > 64)   status = 3;

            const size_t n_ids = (size_t) n_topk * n_tok;
            const size_t n_hid = (size_t) c_embd * n_tok;
            payload.resize(n_ids * (sizeof(int32_t) + sizeof(float)) + n_hid * sizeof(float));
            if (!srv_recv_all(cs, payload.data(), payload.size())) break;
            double t1 = now_ms();

            if (status != 0) {
                fprintf(stderr, "expert-server: rejecting call (layer %d, n_tok %d, k %d, n_embd %d): status %d\n",
                        layer, n_tok, n_topk, c_embd, status);
                int32_t rhdr[4] = { LLAMA_EXPERTS_MAGIC_RET, layer, n_tok, status };
                if (!srv_send_all(cs, rhdr, sizeof(rhdr))) break;
                continue;
            }

            // ---- mirrored llm_build_moe_ffn compute -------------------------
            const expert_layer & L = layers[layer];

            // context sized for: inputs + up/gate/act [n_ff_exp,k,T] + down/weighted [n_embd,k,T]
            // (GPU mode allocates tensor data through gallocr instead - metadata only)
            const size_t need = use_gpu
                ? (size_t) 512 * 1024 + 128 * ggml_tensor_overhead() + ggml_graph_overhead()
                : n_hid * sizeof(float)                                   // hidden
                + n_ids * (sizeof(int32_t) + sizeof(float))               // ids + weights
                + 4 * (size_t) n_ff_exp * n_ids * sizeof(float)           // up, gate, silu, act
                + 2 * (size_t) c_embd   * n_ids * sizeof(float)           // down, weighted
                + (size_t) (n_topk + 4) * n_hid * sizeof(float)           // adds chain slack
                + (size_t) 512 * 1024 + 128 * ggml_tensor_overhead();

            ggml_init_params ip = { need, nullptr, /*no_alloc*/ use_gpu };
            ggml_context * ctx = ggml_init(ip);

            ggml_tensor * hidden  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c_embd, 1, n_tok);
            ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_topk, n_tok);
            ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_topk, n_tok);
            ggml_set_input(hidden); ggml_set_input(ids); ggml_set_input(weights);

            ggml_tensor * moe_out;
            if (form == MOE_FORM_INKLING) {
                // op-for-op mirror of build_inkling's routed-expert tail
                // (src/graphs/build_inkling.cpp): same ops, same order, same
                // shapes, so the same CPU kernels produce the same bytes.
                ggml_tensor * gate = ggml_mul_mat_id(ctx, L.gate_exps, hidden, ids);  // [n_ff, k, T]
                ggml_tensor * up   = ggml_mul_mat_id(ctx, L.up_exps,   hidden, ids);  // [n_ff, k, T]
                ggml_tensor * h    = ggml_mul(ctx, ggml_silu(ctx, gate), up);          // [n_ff, k, T]

                ggml_tensor * experts = ggml_mul_mat_id(ctx, L.down_exps, h, ids);    // [n_embd, k, T]
                experts = ggml_mul(ctx, experts, weights);                            // [n_embd, k, T]

                moe_out = nullptr;
                for (int i = 0; i < n_topk; ++i) {
                    ggml_tensor * e = ggml_view_2d(ctx, experts, c_embd, n_tok,
                            experts->nb[2], (size_t) i * experts->nb[1]);
                    moe_out = moe_out ? ggml_add(ctx, moe_out, e) : e;
                }
                if (!ggml_is_contiguous(moe_out)) {
                    // k == 1 leaves a strided view; the reply is a flat [n_embd, T]
                    moe_out = ggml_cont(ctx, moe_out);
                }
            } else {
            // op-for-op mirror of llm_build_moe_ffn's routed-expert tail
            ggml_tensor * par;
            if (use_fmoe) {
                par = ggml_moe_up_gate(ctx, L.up_exps, L.gate_exps, hidden, ids, GGML_UNARY_OP_SILU);
            } else {
                ggml_tensor * up   = ggml_mul_mat_id(ctx, L.up_exps,   hidden, ids); // [n_ff, k, T]
                ggml_tensor * gate = ggml_mul_mat_id(ctx, L.gate_exps, hidden, ids); // [n_ff, k, T]
                par = ggml_fused_mul_unary(ctx, gate, up, GGML_UNARY_OP_SILU);       // [n_ff, k, T]
            }
            *((float *)(par->op_params + 1)) = swiglu_limit;

            ggml_tensor * down = ggml_mul_mat_id(ctx, L.down_exps, par, ids);        // [n_embd, k, T]

            if (use_mmad) {
                moe_out = ggml_mul_multi_add(ctx, down, weights);                    // [n_embd, T]
            } else {
                ggml_tensor * wexp = ggml_mul(ctx, down, weights);                   // [n_embd, k, T]
                moe_out = ggml_multi_add(ctx,
                        ggml_view_2d(ctx, wexp, c_embd, n_tok, wexp->nb[2], 0), n_topk);
            }
            }
            ggml_set_output(moe_out);

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, moe_out);

            const uint8_t * r = payload.data();
            bool ok = true;
            if (use_gpu) {
                if (!ggml_gallocr_alloc_graph(conn_galloc, gf)) {
                    fprintf(stderr, "expert-server: gallocr failed (n_tok %d)\n", n_tok);
                    ggml_free(ctx);
                    break;
                }
                ggml_backend_tensor_set(ids,     r, 0, n_ids * sizeof(int32_t)); r += n_ids * sizeof(int32_t);
                ggml_backend_tensor_set(weights, r, 0, n_ids * sizeof(float));   r += n_ids * sizeof(float);
                ggml_backend_tensor_set(hidden,  r, 0, n_hid * sizeof(float));

                ggml_backend_graph_compute(gbackend, gf);

                if (outbuf.size() < n_hid * sizeof(float)) outbuf.resize(n_hid * sizeof(float));
                ggml_backend_tensor_get(moe_out, outbuf.data(), 0, n_hid * sizeof(float));
                double t2g = now_ms();

                int32_t rhdr[4] = { LLAMA_EXPERTS_MAGIC_RET, layer, n_tok, 0 };
                ok = srv_send_all(cs, rhdr, sizeof(rhdr)) &&
                     srv_send_all(cs, outbuf.data(), n_hid * sizeof(float));
                ggml_free(ctx);
                if (!ok) break;
                double t3g = now_ms();
                n_calls++;
                t_recv += t1 - t0; t_comp += t2g - t1; t_send += t3g - t2g;
                {
                    srv_phase & P = (n_tok == 1) ? ph_dec : ph_pre;
                    P.n++; P.rows += (uint64_t) n_tok;
                    P.idle += idle_ms; P.recv += t1 - t0; P.comp += t2g - t1; P.send += t3g - t2g;
                }
                t_prev_done = t3g;
                if ((n_calls % 1024) == 0) {
                    fprintf(stderr, "expert-server: %" PRIu64 " calls, avg recv %.3f ms comp %.3f ms send %.3f ms\n",
                            n_calls, t_recv / n_calls, t_comp / n_calls, t_send / n_calls);
                }
                continue;
            }

            memcpy(ids->data,     r, n_ids * sizeof(int32_t)); r += n_ids * sizeof(int32_t);
            memcpy(weights->data, r, n_ids * sizeof(float));   r += n_ids * sizeof(float);
            memcpy(hidden->data,  r, n_hid * sizeof(float));

            ggml_cplan plan = ggml_graph_plan(gf, n_threads);
            if (plan.work_size > 0) {
                if (workbuf.size() < plan.work_size) workbuf.resize(plan.work_size);
                plan.work_data = workbuf.data();
            }
            ggml_graph_compute(gf, &plan);
            double t2 = now_ms();

            int32_t rhdr[4] = { LLAMA_EXPERTS_MAGIC_RET, layer, n_tok, 0 };
            ok = srv_send_all(cs, rhdr, sizeof(rhdr));
            if (ok) {
                // both reduction forms produce a contiguous [n_embd, n_tok] result
                ok = srv_send_all(cs, moe_out->data, n_hid * sizeof(float));
            }
            ggml_free(ctx);
            if (!ok) break;
            double t3 = now_ms();

            n_calls++;
            t_recv += t1 - t0; t_comp += t2 - t1; t_send += t3 - t2;
            {
                srv_phase & P = (n_tok == 1) ? ph_dec : ph_pre;
                P.n++; P.rows += (uint64_t) n_tok;
                P.idle += idle_ms; P.recv += t1 - t0; P.comp += t2 - t1; P.send += t3 - t2;
            }
            t_prev_done = t3;
            if ((n_calls % 1024) == 0) {
                fprintf(stderr, "expert-server: %" PRIu64 " calls, avg recv %.3f ms comp %.3f ms send %.3f ms\n",
                        n_calls, t_recv / n_calls, t_comp / n_calls, t_send / n_calls);
            }
        }

        close(cs);
        if (n_calls > 0) {
            fprintf(stderr, "expert-server: client done: %" PRIu64 " calls, avg recv %.3f ms comp %.3f ms send %.3f ms\n",
                    n_calls, t_recv / n_calls, t_comp / n_calls, t_send / n_calls);
            for (int i = 0; i < 2; ++i) {
                const srv_phase & P = i ? ph_pre : ph_dec;
                if (P.n == 0) continue;
                const double n = (double) P.n;
                fprintf(stderr, "expert-server: %-7s %6" PRIu64 " calls %9" PRIu64 " rows | idle %8.3f  recv %8.3f  comp %8.3f  send %8.3f ms/call"
                                " | busy %.1f s of %.1f s wall\n",
                        i ? "prefill" : "decode", P.n, P.rows,
                        P.idle / n, P.recv / n, P.comp / n, P.send / n,
                        (P.recv + P.comp + P.send) / 1000.0,
                        (P.idle + P.recv + P.comp + P.send) / 1000.0);
            }
        } else {
            fprintf(stderr, "expert-server: client disconnected\n");
        }
        if (conn_galloc) ggml_gallocr_free(conn_galloc);
        }).detach();
    }
}
