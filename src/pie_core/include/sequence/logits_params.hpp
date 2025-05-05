#pragma once

#include <cstdint>
#include <mlx/mlx.h>
#include <unordered_map>

namespace mx = mlx::core;

namespace pie_core {

    struct LogitsParams {
        float frequency_penalty = 0.0f;
        std::unordered_map<int32_t, float> logit_bias;
        float presence_penalty = 0.0f;
        int repetition_context_size = 60;
        float repetition_penalty = 1.0f;
    };

} // namespace pie_core
