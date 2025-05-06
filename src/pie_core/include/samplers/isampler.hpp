#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <random>
#include <vector>
#include <memory>

#include "sequence/sampling_params.hpp"

namespace mx = mlx::core;

namespace pie_core::samplers {

    // Abstract base class for all samplers.
    class ISampler {
    public:
        virtual ~ISampler() = default;

        // Core sampling method.
        virtual mx::array next_token(
            const mx::array& logits,
            const sequence::SamplingParams& params,
            std::mt19937& rng
        ) = 0;
    };

} // namespace pie_core::samplers
