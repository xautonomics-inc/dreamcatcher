#pragma once

// Layer-library manifest parsing. The implementation moved to
// common/layer-library.h so llama-server / llama-cli (--model-dir) share one
// enumeration with llama-stage-runner; these names are kept as the stage-side
// spelling used by the runner and tests/test-stage-manifest.cpp.

#include "../../common/layer-library.h"

#include <cstdint>
#include <istream>
#include <string>

inline int32_t stage_parse_source_block_count(std::istream & input) {
    return layer_library::parse_source_block_count(input);
}

inline int32_t stage_read_source_block_count(const std::string & dir) {
    return layer_library::read_source_block_count(dir);
}
