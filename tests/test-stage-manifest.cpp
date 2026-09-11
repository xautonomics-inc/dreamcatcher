#include "../examples/stage-runner/layer-manifest.h"

#include <iostream>
#include <sstream>

int main() {
    // Files may precede source, use compact formatting, or contain decoy counts.
    const char * valid[] = {
        R"({"files":[{"window_kv":{"block_count":1}}],"source":{"block_count":43}})",
        "{\n \"source\": {\n \"block_count\": 43\n }, \"block_count\": 15\n}",
        R"({"source":{"block_count":43},"description":"\"block_count\":1"})",
    };
    for (const auto * fixture : valid) {
        std::istringstream input(fixture);
        if (stage_parse_source_block_count(input) != 43) return 1;
    }
    const char * invalid[] = {
        "", "{", "null", "[]", "{}", R"({"block_count":43})",
        R"({"source":{}})", R"({"source":{"block_count":0}})",
        R"({"source":{"block_count":-1}})", R"({"source":{"block_count":43.0}})",
        R"({"source":{"block_count":"43"}})", R"({"source":{"block_count":true}})",
        R"({"source":{"block_count":null}})", R"({"source":{"block_count":2147483648}})",
        R"({"source":{"block_count":18446744073709551615}})",
        R"({"source":{"block_count":43}} trailing)",
    };
    for (const auto * fixture : invalid) {
        bool rejected = false;
        try {
            std::istringstream input(fixture);
            stage_parse_source_block_count(input);
        } catch (const std::exception &) {
            rejected = true;
        }
        if (!rejected) {
            std::cerr << "accepted invalid manifest: " << fixture << '\n';
            return 1;
        }
    }
    std::istringstream largest(R"({"source":{"block_count":2147483647}})");
    if (stage_parse_source_block_count(largest) != 2147483647) return 1;
    std::cout << "stage manifest: 20 cases passed\n";
}
