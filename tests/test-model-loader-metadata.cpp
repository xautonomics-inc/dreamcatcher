#include "llama-model-loader.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    gguf_context * meta = gguf_init_empty();

    const std::vector<uint32_t> per_block_u32 = { 10, 20, 30, 40 };
    const std::vector<float> per_block_f32 = { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<const char *> per_block_str = { "zero", "one", "two", "three" };
    const std::vector<int16_t> unrelated = { 1, 2, 3 };
    gguf_set_arr_data(meta, "test.per_block_u32", GGUF_TYPE_UINT32, per_block_u32.data(), per_block_u32.size());
    gguf_set_arr_data(meta, "test.per_block_f32", GGUF_TYPE_FLOAT32, per_block_f32.data(), per_block_f32.size());
    gguf_set_arr_str(meta, "test.per_block_str", per_block_str.data(), per_block_str.size());
    gguf_set_arr_data(meta, "test.unrelated", GGUF_TYPE_INT16, unrelated.data(), unrelated.size());
    gguf_set_val_u32(meta, "test.scalar", 4);

    llama_model_loader_slice_block_arrays(meta, 4, 1, 2);

    int kid = gguf_find_key(meta, "test.per_block_u32");
    assert(gguf_get_arr_n(meta, kid) == 2);
    const auto * u32 = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
    assert(u32[0] == 20 && u32[1] == 30);

    kid = gguf_find_key(meta, "test.per_block_f32");
    assert(gguf_get_arr_n(meta, kid) == 2);
    const auto * f32 = static_cast<const float *>(gguf_get_arr_data(meta, kid));
    assert(f32[0] == 2.0f && f32[1] == 3.0f);

    kid = gguf_find_key(meta, "test.per_block_str");
    assert(gguf_get_arr_n(meta, kid) == 2);
    assert(std::string(gguf_get_arr_str(meta, kid, 0)) == "one");
    assert(std::string(gguf_get_arr_str(meta, kid, 1)) == "two");

    kid = gguf_find_key(meta, "test.unrelated");
    assert(gguf_get_arr_n(meta, kid) == 3);
    assert(gguf_get_val_u32(meta, gguf_find_key(meta, "test.scalar")) == 4);

    bool rejected = false;
    try {
        llama_model_loader_slice_block_arrays(meta, 4, 3, 2);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    gguf_free(meta);
    return 0;
}
