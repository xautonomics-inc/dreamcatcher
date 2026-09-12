#pragma once

// =====================================================================
// ik-port compatibility layer for the grafted 2026 mainline Vulkan backend
//
// Lets the 2026 mainline ggml Vulkan backend (ggml-vulkan.cpp, grafted
// from the fleet llama.cpp fork @ glm52-ring-mmid-batched) build against
// ik_llama.cpp's older ggml core:
//
//  - logging macros            (ik core has no GGML_LOG_*)
//  - graph-fusion helpers      (ggml_can_fuse & friends, ported from
//    mainline ggml-impl.h/ggml.c; ik's ggml_cgraph has no use_counts
//    field, so a per-submission use-count map is maintained instead —
//    ggml_vk_compat_refresh_use_counts() MUST be called at the top of
//    ggml_backend_vk_graph_compute before any fusion query)
//
// Only ggml-vulkan.cpp may include this header.
// =====================================================================

#include "ggml.h"
#include "ggml-impl.h"

#include <cstdint>
#include <cstdio>
#include <array>
#include <initializer_list>
#include <unordered_map>

// ---------------------------------------------------------------------
// logging (mainline ggml has GGML_LOG_* wired to a callback; ik does not)
// ---------------------------------------------------------------------
#ifndef GGML_LOG_WARN
#define GGML_LOG_INFO(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_WARN(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_CONT(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) do { } while (0)
#endif

// ---------------------------------------------------------------------
// per-submission use counts
// ---------------------------------------------------------------------
struct ggml_vk_compat_use_counts_t {
    const struct ggml_cgraph * graph = nullptr;
    std::unordered_map<const struct ggml_tensor *, int32_t> counts;
};

static thread_local ggml_vk_compat_use_counts_t ggml_vk_compat_use_counts;

// Rebuild the use-count map for this graph submission. The sched reuses
// cgraph storage across submissions, so this must be called every time.
static void ggml_vk_compat_refresh_use_counts(const struct ggml_cgraph * cgraph) {
    auto & uc = ggml_vk_compat_use_counts;
    uc.graph = cgraph;
    uc.counts.clear();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s]) {
                uc.counts[node->src[s]]++;
            }
        }
    }
}

static inline int32_t ggml_node_get_use_count(const struct ggml_cgraph * cgraph, int node_idx) {
    const auto & uc = ggml_vk_compat_use_counts;
    // fusion queries must only happen on the graph the map was built for
    if (uc.graph != cgraph) {
        // unknown graph: report a safe over-estimate so no fusion happens
        return INT32_MAX;
    }
    auto it = uc.counts.find(cgraph->nodes[node_idx]);
    return it == uc.counts.end() ? 0 : it->second;
}

// ---------------------------------------------------------------------
// fusion helpers (verbatim mainline logic, minus GGML_TENSOR_FLAG_COMPUTE
// which does not exist in ik: every node in an ik graph is computed)
// ---------------------------------------------------------------------
static inline bool ggml_vk_compat_node_is_compute(const struct ggml_tensor * node) {
    (void) node;
    return true; // ik has no GGML_TENSOR_FLAG_COMPUTE; all graph nodes are computed
}

static bool ggml_op_is_empty(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static inline bool ggml_node_has_n_uses(const struct ggml_cgraph * cgraph, int node_idx, int32_t n_uses) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];

    // check the use count against how many we're replacing
    if (ggml_node_get_use_count(cgraph, node_idx) != n_uses) {
        return false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    if (node->view_src) {
        return false;
    }

    // If the user requested output for the node, can't fuse
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }

    return true;
}

static inline bool ggml_can_fuse_ext(const struct ggml_cgraph * cgraph, const int * node_idxs, const enum ggml_op * ops, int num_ops) {
    for (int i = 0; i < num_ops; ++i) {
        if (node_idxs[i] >= cgraph->n_nodes) {
            return false;
        }

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];
        if (node->op != ops[i]) {
            return false;
        }
        if (!ggml_vk_compat_node_is_compute(node)) {
            return false;
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idxs[i], 1)) {
            return false;
        }
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;
            }
            if (!ggml_are_same_shape(node, prev)) {
                return false;
            }
        }
    }
    return true;
}

// same as above, for sequential indices starting at node_idx
static inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, const enum ggml_op * ops, int num_ops) {
    GGML_ASSERT(num_ops < 32);

    if (node_idx + num_ops > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];
    for (int i = 0; i < num_ops; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_ext(cgraph, idxs, ops, num_ops);
}

// find tensor in an index list; returns position or -1
static inline int ggml_node_list_find_tensor(const struct ggml_cgraph * cgraph,
                                             const int * idxs, int count,
                                             const struct ggml_tensor * tensor) {
    for (int i = 0; i < count; ++i) {
        if (cgraph->nodes[idxs[i]] == tensor) {
            return i;
        }
    }
    return -1;
}

static inline bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph * cgraph,
                                              const int *                node_idxs,
                                              int                        count,
                                              const enum ggml_op *       ops,
                                              const int *                outputs,
                                              int                        num_outputs) {
    GGML_ASSERT(outputs && num_outputs > 0);

    for (int i = 0; i < count; ++i) {
        if (node_idxs[i] >= cgraph->n_nodes) {
            return false;
        }

        const struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];

        if (node->op != ops[i]) {
            return false;
        }

        if (!ggml_vk_compat_node_is_compute(node)) {
            return false;
        }

        if (ggml_node_list_find_tensor(cgraph, outputs, num_outputs, node) != -1) {
            continue;
        }

        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }

        int subgraph_uses = 0;
        for (int j = i + 1; j < count; ++j) {
            const struct ggml_tensor * other_node = cgraph->nodes[node_idxs[j]];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                if (other_node->src[src_idx] == node) {
                    subgraph_uses++;
                }
            }
        }

        if (subgraph_uses != ggml_node_get_use_count(cgraph, node_idxs[i])) {
            return false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph
        struct ggml_tensor * view_src = node->view_src;
        while (view_src) {
            if (ggml_node_list_find_tensor(cgraph, node_idxs, count, view_src) == -1) {
                return false;
            }
            view_src = view_src->view_src;
        }
    }

    return true;
}

static inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph * cgraph,
                                          int                        node_idx,
                                          int                        count,
                                          const enum ggml_op *       ops,
                                          const int *                outputs,
                                          int                        num_outputs) {
    GGML_ASSERT(count < 32);
    if (node_idx + count > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];

    for (int i = 0; i < count; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outputs, num_outputs);
}

// nicer C++ syntax for ggml_can_fuse
inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
}

inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *          cgraph,
                                   int                                 start_idx,
                                   std::initializer_list<enum ggml_op> ops,
                                   std::initializer_list<int>          outputs = {}) {
    return ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
inline bool ggml_check_edges(const struct ggml_cgraph *                cgraph,
                             int                                       start_idx,
                             std::initializer_list<std::array<int, 3>> edges) {
    for (const auto & edge : edges) {
        int dst_node = edge[0];
        int src_idx  = edge[1];
        int src_node = edge[2];
        if (cgraph->nodes[start_idx + dst_node]->src[src_idx] != cgraph->nodes[start_idx + src_node]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// type aliases: mainline GGML_TYPE_Q1_0 (id 41, fp16 scale + 128x1bit) is
// byte-identical to ik's GGML_TYPE_Q1_0_G128 (same id, same block layout)
// ---------------------------------------------------------------------
#define GGML_TYPE_Q1_0 GGML_TYPE_Q1_0_G128

// ---------------------------------------------------------------------
// small enum/flag gaps vs 2026 mainline ggml.h
// ---------------------------------------------------------------------
#ifndef GGML_ROPE_TYPE_NORMAL
#define GGML_ROPE_TYPE_NORMAL 0
#endif
// ik only has GGML_SCALE_FLAG_ALIGN_CORNERS (1<<8); mainline adds ANTIALIAS.
// ik graphs never set it, so the antialias upscale path is simply never taken.
#ifndef GGML_SCALE_FLAG_ANTIALIAS
#define GGML_SCALE_FLAG_ANTIALIAS (1 << 9)
#endif
