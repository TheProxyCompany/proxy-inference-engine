#include "samplers/categorical.hpp"
#include "samplers/sampler_registry.hpp"

#include <mlx/ops.h>
#include <mlx/random.h>

namespace mx = mlx::core;

namespace pie_core::samplers {

    mx::array CategoricalSampler::next_token(
        const mx::array& logits,
        const sequence::SamplingParams&,
        std::mt19937&
    ) {
        return mx::random::categorical(logits, /*axis=*/-1, /*keepdims=*/false);
    }

    namespace {
        SamplerRegistrar<CategoricalSampler> registrar("categorical");
    }
}
