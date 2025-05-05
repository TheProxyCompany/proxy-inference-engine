#pragma once

#include <cstdint>

namespace pie_core {

    struct SamplingParams {
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = -1;
        float min_p = 0.0f;
        uint32_t rng_seed;
    };

} // namespace pie_core
