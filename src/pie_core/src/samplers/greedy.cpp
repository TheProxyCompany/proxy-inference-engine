#include "samplers/greedy.hpp"
#include "samplers/sampler_registry.hpp"

#include <mlx/ops.h>

namespace pie_core::samplers {

    mx::array GreedySampler::next_token(
        const mx::array& logits,
        const sequence::SamplingParams&,
        std::mt19937&
    ) {
        return mx::argmax(logits, /*axis=*/-1, /*keepdims=*/false);
    }

    namespace {
        SamplerRegistrar<GreedySampler> registrar("greedy");
    }
}
