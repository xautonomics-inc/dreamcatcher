#include "llama-model-loader.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main() {
    gguf_context * meta = gguf_init_empty();

    const std::vector<uint32_t> per_block_u32 = { 10, 20, 30, 40 };
    const std::vector<float> per_block_f32 = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<const char *> per_block_str = { "zero", "one", "two", "three" };
    const std::vector<int16_t> unrelated = { 1, 2, 3 };
    gguf_set_arr_data(meta, "test.per_block_u32", GGUF_TYPE_UINT32, per_block_u32.data(), per_block_u32.size());
    gguf_set_arr_data(meta, "test.per_block_f32", GGUF_TYPE_FLOAT32, per_block_f32.data(), per_block_f32.size());
    gguf_set_arr_str(meta, "test.per_block_str", per_block_str.data(), per_block_str.size());
    gguf_set_arr_data(meta, "test.unrelated", GGUF_TYPE_INT16, unrelated.data(), unrelated.size());
    gguf_set_val_u32(meta, "test.scalar", 4);

    llama_model_loader_slice_block_arrays(meta, 4, 1, 2);

    int kid = gguf_find_key(meta, "test.per_block_u32");
    CHECK(gguf_get_arr_n(meta, kid) == 2);
    const auto * u32 = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
    CHECK(u32[0] == 20 && u32[1] == 30);

    kid = gguf_find_key(meta, "test.per_block_f32");
    CHECK(gguf_get_arr_n(meta, kid) == 2);
    const auto * f32 = static_cast<const float *>(gguf_get_arr_data(meta, kid));
    CHECK(f32[0] == 2.0f && f32[1] == 3.0f);

    kid = gguf_find_key(meta, "test.per_block_str");
    CHECK(gguf_get_arr_n(meta, kid) == 2);
    CHECK(std::string(gguf_get_arr_str(meta, kid, 0)) == "one");
    CHECK(std::string(gguf_get_arr_str(meta, kid, 1)) == "two");

    kid = gguf_find_key(meta, "test.unrelated");
    CHECK(gguf_get_arr_n(meta, kid) == 3);
    CHECK(gguf_get_val_u32(meta, gguf_find_key(meta, "test.scalar")) == 4);

    bool rejected = false;
    try {
        llama_model_loader_slice_block_arrays(meta, 4, 3, 2);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);

    gguf_free(meta);

    // A per-layer array can be LONGER than block_count: DSV4 copies attention.compress_ratios
    // out of the upstream config, which covers the blocks plus trailing companion slots. Its
    // block prefix still has to follow the window, or every layer of a window that does not
    // open at block 0 is built with another layer's compression class. hash_layer_count counts
    // a LEADING RUN of blocks and has to be shifted onto the window for the same reason.
    const std::vector<uint32_t> ratios     = { 0, 0, 4, 128, 4, 128, 0, 0 };  // 6 blocks + 2 extra
    const std::vector<uint32_t> per_block_6 = { 9, 8, 7, 6, 5, 4 };           // plain per-block array
    const std::vector<uint32_t> long_other  = { 1, 2, 3, 4, 5, 6, 7, 8 };     // long, not per-layer

    struct window_case {
        int32_t  start;
        int32_t  count;
        uint32_t ratio_first;
        uint32_t ratio_last;
        uint32_t hash_layers;
    };
    const window_case cases[] = {
        { 0, 6, 0, 128, 3 },  // the whole model: nothing moves
        { 2, 2, 4, 128, 1 },  // [2,4): one hash layer left (block 2), CSA then HCA
        { 4, 2, 4, 128, 0 },  // [4,6): past the hash run entirely
    };

    for (const auto & c : cases) {
        gguf_context * win = gguf_init_empty();
        gguf_set_arr_data(win, "deepseek4.attention.compress_ratios", GGUF_TYPE_UINT32, ratios.data(), ratios.size());
        gguf_set_arr_data(win, "test.per_block_6", GGUF_TYPE_UINT32, per_block_6.data(), per_block_6.size());
        gguf_set_arr_data(win, "test.long_not_per_layer", GGUF_TYPE_UINT32, long_other.data(), long_other.size());
        gguf_set_val_u32(win, "deepseek4.hash_layer_count", 3);

        llama_model_loader_slice_block_arrays(win, 6, c.start, c.count);

        int wid = gguf_find_key(win, "deepseek4.attention.compress_ratios");
        CHECK(gguf_get_arr_n(win, wid) == c.count);
        const auto * win_ratios = static_cast<const uint32_t *>(gguf_get_arr_data(win, wid));
        CHECK(win_ratios[0] == c.ratio_first);
        CHECK(win_ratios[c.count - 1] == c.ratio_last);

        CHECK(gguf_get_val_u32(win, gguf_find_key(win, "deepseek4.hash_layer_count")) == c.hash_layers);

        // a plain per-block array still follows the window; a long array that is not a known
        // per-layer key is NOT sliced - only the allowlisted ones may exceed block_count
        wid = gguf_find_key(win, "test.per_block_6");
        CHECK(gguf_get_arr_n(win, wid) == c.count);
        CHECK(static_cast<const uint32_t *>(gguf_get_arr_data(win, wid))[0] == per_block_6[(size_t) c.start]);

        wid = gguf_find_key(win, "test.long_not_per_layer");
        CHECK(gguf_get_arr_n(win, wid) == (int) long_other.size());

        gguf_free(win);
    }

    // a per-layer key SHORTER than block_count is malformed metadata, not a window: left alone
    gguf_context * odd = gguf_init_empty();
    const std::vector<uint32_t> short_ratios = { 1, 2 };
    gguf_set_arr_data(odd, "deepseek4.attention.compress_ratios", GGUF_TYPE_UINT32, short_ratios.data(), short_ratios.size());
    llama_model_loader_slice_block_arrays(odd, 6, 2, 2);
    const int oid = gguf_find_key(odd, "deepseek4.attention.compress_ratios");
    CHECK(gguf_get_arr_n(odd, oid) == 2);
    const auto * kept = static_cast<const uint32_t *>(gguf_get_arr_data(odd, oid));
    CHECK(kept[0] == 1 && kept[1] == 2);
    gguf_free(odd);

    printf("model loader metadata: window slicing, long per-layer arrays and leading-run counts pass\n");
    return 0;
}
