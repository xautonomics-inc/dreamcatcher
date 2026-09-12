#pragma once

// Layer library ("--model-dir"): enumerate a per-layer GGUF model library and
// compose the part list that llama_model_load_from_parts() (include/llama.h)
// assembles into one logical model.
//
// A "layer library" is a directory produced by slice_gguf_layers.py:
//   blk-NNNNN.gguf          one transformer layer (abs index NNNNN), tensors blk.0.*
//   parts-embd.gguf         token_embd.weight
//   parts-output.gguf       output_norm.weight (+ output.weight if untied)
//   parts-nextn-NNNNN.gguf  one NextN/MTP block (abs index NNNNN), tensors blk.0.*
//   parts-other.gguf        any other non-blk tensors (rare; e.g. rope_freqs)
//   manifest.json           source.block_count + provenance + per-tensor hashes
//
// Shared by two callers with identical file-selection rules:
//   - llama-stage-runner, which loads ONE layer window per pipeline stage;
//   - the common loader behind llama-server / llama-cli --model-dir, which loads
//     the whole library in a single process (window default = the full model).
// `tag` prefixes every diagnostic so each caller keeps its own message prefix.

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace layer_library {

// One input file of an assembly; mirrors llama_model_part with owned storage.
struct part {
    std::string path;
    int32_t     blk_base;
    int32_t     source_blk_start;
    int32_t     source_blk_count;
};

// Part GGUF block_count is 0 or 1, and file names describe only the local subset.
// Only source.block_count in the original manifest describes the unsliced model.
inline int32_t parse_source_block_count(std::istream & input) {
    const auto manifest = nlohmann::json::parse(input);
    const auto & value = manifest.at("source").at("block_count");
    if (!value.is_number_integer()) {
        throw std::runtime_error("manifest source.block_count must be a positive int32");
    }
    const auto max_count = std::numeric_limits<int32_t>::max();
    if ((value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(max_count)) ||
        (!value.is_number_unsigned() && value.get<int64_t>() <= 0) ||
        value.get<int64_t>() > max_count || value.get<int64_t>() == 0) {
        throw std::runtime_error("manifest source.block_count must be a positive int32");
    }
    return value.get<int32_t>();
}

inline int32_t read_source_block_count(const std::string & dir) {
    std::ifstream input(dir + "/manifest.json");
    if (!input) {
        throw std::runtime_error("cannot read manifest.json; copy the original library manifest with the window parts");
    }
    return parse_source_block_count(input);
}

// "A,B" -> window [A,B). Returns false on a malformed spec.
inline bool parse_window(const std::string & spec, int & a, int & b) {
    const size_t c = spec.find(',');
    if (c == std::string::npos || c == 0 || c + 1 >= spec.size()) {
        return false;
    }
    a = atoi(spec.substr(0, c).c_str());
    b = atoi(spec.substr(c + 1).c_str());
    return true;
}

// Compose the part list for the absolute window [win_a, win_b) into `dir`.
// Negative bounds take the default: the FULL model ([0, source.block_count)).
// parts_spec: "" / "auto" | "none" | a comma list of embd,output,nextn,other.
// Returns false (after one diagnostic on stderr) if the library cannot serve it.
inline bool assemble(const std::string & dir, int win_a, int win_b,
                     const std::string & parts_spec, std::vector<part> & out,
                     const char * tag = "layer-library") {
    std::map<int, std::string> layers, nextn;   // abs index -> filename
    std::string f_embd, f_output, f_other;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) { fprintf(stderr, "%s: cannot open --model-dir %s\n", tag, dir.c_str()); return false; }
    for (const auto & entry : it) {
        const std::string name = entry.path().filename().string();
        int idx; char tail8[8] = {0};
        if      (sscanf(name.c_str(), "blk-%d.ggu%1s",         &idx, tail8) == 2 && !strcmp(tail8, "f")) layers[idx] = name;
        else if (sscanf(name.c_str(), "parts-nextn-%d.ggu%1s", &idx, tail8) == 2 && !strcmp(tail8, "f")) nextn[idx]  = name;
        else if (name == "parts-embd.gguf")   f_embd   = name;
        else if (name == "parts-output.gguf") f_output = name;
        else if (name == "parts-other.gguf")  f_other  = name;
    }
    if (layers.empty() && nextn.empty()) { fprintf(stderr, "%s: no blk-*.gguf in %s\n", tag, dir.c_str()); return false; }
    int32_t source_blk_count = 0;
    try {
        source_blk_count = read_source_block_count(dir);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: library %s: %s; refusing assembly\n", tag, dir.c_str(), e.what());
        return false;
    }
    const int highest_present = std::max(
        layers.empty() ? -1 : layers.rbegin()->first,
        nextn.empty() ? -1 : nextn.rbegin()->first);
    if (highest_present >= source_blk_count) {
        fprintf(stderr, "%s: library %s has blk.%d beyond source block_count=%d\n",
                tag, dir.c_str(), highest_present, source_blk_count);
        return false;
    }
    const int n_total = source_blk_count;
    if (win_a < 0) win_a = 0;
    if (win_b < 0) win_b = n_total;   // default: full model
    if (win_a >= win_b || win_b > n_total) {
        fprintf(stderr, "%s: bad --layers window [%d,%d) (library has %d blocks)\n", tag, win_a, win_b, n_total);
        return false;
    }
    // parts selection: "auto" mirrors today's monolithic slices (embd+output in EVERY
    // slice; nextn blocks iff the window covers them); "none" = bare layers (NOTE:
    // most archs require token_embd at load); or an explicit comma list.
    bool inc_embd = false, inc_output = false, inc_nextn = false, inc_other = false;
    if (parts_spec.empty() || parts_spec == "auto") {
        inc_embd = !f_embd.empty(); inc_output = !f_output.empty(); inc_nextn = true;
        inc_other = !f_other.empty();
        if (inc_other) fprintf(stderr, "%s: including parts-other.gguf (present in library)\n", tag);
    } else if (parts_spec != "none") {
        std::string s = parts_spec; size_t p = 0;
        while (p <= s.size()) {
            size_t c = s.find(',', p);
            std::string one = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if      (one == "embd")   inc_embd   = true;
            else if (one == "output") inc_output = true;
            else if (one == "nextn")  inc_nextn  = true;
            else if (one == "other")  inc_other  = true;
            else if (!one.empty()) { fprintf(stderr, "%s: unknown --stage-parts item '%s'\n", tag, one.c_str()); return false; }
            if (c == std::string::npos) break; p = c + 1;
        }
    }
    if (inc_embd   && f_embd.empty())   { fprintf(stderr, "%s: parts-embd.gguf not in library\n",   tag); return false; }
    if (inc_output && f_output.empty()) { fprintf(stderr, "%s: parts-output.gguf not in library\n", tag); return false; }
    if (inc_other  && f_other.empty())  { fprintf(stderr, "%s: parts-other.gguf not in library\n",  tag); return false; }
    if (inc_embd) out.push_back({ dir + "/" + f_embd, 0, win_a, n_total });
    for (int i = win_a; i < win_b; ++i) {
        auto li = layers.find(i);
        if (li != layers.end()) { out.push_back({ dir + "/" + li->second, i - win_a, win_a, n_total }); continue; }
        auto ni = nextn.find(i);
        if (ni != nextn.end()) {
            if (!inc_nextn) { fprintf(stderr, "%s: window [%d,%d) covers NextN blk %d but --stage-parts excludes nextn\n", tag, win_a, win_b, i); return false; }
            out.push_back({ dir + "/" + ni->second, i - win_a, win_a, n_total });
            continue;
        }
        fprintf(stderr, "%s: library %s has no file for blk %d (window [%d,%d))\n", tag, dir.c_str(), i, win_a, win_b);
        return false;
    }
    if (inc_output) out.push_back({ dir + "/" + f_output, 0, win_a, n_total });
    if (inc_other)  out.push_back({ dir + "/" + f_other,  0, win_a, n_total });
    fprintf(stderr, "%s: assembling window [%d,%d) from %zu part files in %s (embd=%d output=%d other=%d)\n",
            tag, win_a, win_b, out.size(), dir.c_str(), (int) inc_embd, (int) inc_output, (int) inc_other);
    return true;
}

} // namespace layer_library
