#include "samplers/sampler_factory.hpp"
#include "samplers/sampler_registry.hpp"
#include "sequence/sampling_params.hpp"
#include <string>
#include <memory>
#include <iostream>

namespace pie_core::samplers {

    std::unique_ptr<ISampler> create_sampler(const sequence::SamplingParams& params) {
        if (params.temperature == 0.0f) {
            return SamplerRegistry::create_sampler("greedy");
        }
        return SamplerRegistry::create_sampler("categorical");
    }

}
