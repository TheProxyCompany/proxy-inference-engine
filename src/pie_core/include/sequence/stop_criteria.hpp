#pragma once

#include <vector>
#include <cstdint>

namespace pie_core::sequence {

    struct StopCriteria {
        int max_generated_tokens = 1024;
        std::vector<int32_t> stop_token_ids;
    };

} // namespace pie_core
