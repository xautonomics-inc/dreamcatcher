#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

// Part GGUF block_count is 0 or 1, and file names describe only the local subset.
// Only source.block_count in the original manifest describes the unsliced model.
inline int32_t stage_parse_source_block_count(std::istream & input) {
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

inline int32_t stage_read_source_block_count(const std::string & dir) {
    std::ifstream input(dir + "/manifest.json");
    if (!input) {
        throw std::runtime_error("cannot read manifest.json; copy the original library manifest with the window parts");
    }
    return stage_parse_source_block_count(input);
}
