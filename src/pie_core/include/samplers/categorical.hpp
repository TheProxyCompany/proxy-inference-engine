#pragma once

#include "samplers/isampler.hpp"
#include "sequence/sampling_params.hpp"

#include <mlx/array.h>
#include <random>
#include <vector>

namespace mx = mlx::core;

namespace pie_core::samplers {

    // --- Categorical Sampler Implementation ---
    class CategoricalSampler : public ISampler {
        public:
            // Declare the overridden method
            mx::array next_token(
                const mx::array& logits,
                const sequence::SamplingParams& params,
                std::mt19937& rng
            ) override;
    };
}
