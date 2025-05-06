#pragma once

#include "logit_processors/logit_processor.hpp"
#include "sequence/logits_params.hpp"
#include <memory>
#include <vector>

namespace pie_core::logit_processors {

    // Factory function to create the appropriate logit processor based on parameters.
    // Returns a unique_ptr to manage ownership.
    std::unique_ptr<ILogitProcessor> create_processor(const std::string& processor_type);

    // Factory function to create all applicable processors based on logits params
    // Returns a vector of unique_ptrs to all relevant processors
    std::vector<std::unique_ptr<ILogitProcessor>> create_processors(const sequence::LogitsParams& params);

}
