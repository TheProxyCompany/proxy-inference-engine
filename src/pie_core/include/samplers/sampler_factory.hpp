#pragma once

#include "samplers/isampler.hpp"
#include "sequence/sampling_params.hpp"
#include <memory>

namespace pie_core::samplers {

    // Factory function to create the appropriate sampler based on parameters.
    // Returns a unique_ptr to manage ownership.
    std::unique_ptr<ISampler> create_sampler(const sequence::SamplingParams& params);

}
