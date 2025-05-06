#include "logit_processors/logit_processor_factory.hpp"
#include "logit_processors/logit_processor_registry.hpp"
#include "sequence/logits_params.hpp"
#include <memory>
#include <vector>

namespace pie_core::logit_processors {

    std::unique_ptr<ILogitProcessor> create_processor(const std::string& processor_type) {
        return LogitProcessorRegistry::create_processor(processor_type);
    }

    std::vector<std::unique_ptr<ILogitProcessor>> create_processors(const sequence::LogitsParams& params) {
        std::vector<std::unique_ptr<ILogitProcessor>> processors;

        // Add repetition processor if repetition penalty is not 1.0
        if (params.repetition_penalty != 1.0f) {
            processors.push_back(create_processor("repetition"));
        }

        // Add frequency penalty processor if frequency penalty is not 0
        if (params.frequency_penalty != 0.0f) {
            processors.push_back(create_processor("frequency_penalty"));
        }

        // Add presence penalty processor if presence penalty is not 0
        if (params.presence_penalty != 0.0f) {
            processors.push_back(create_processor("presence_penalty"));
        }

        // Add logit bias processor if logit_bias map is not empty
        if (!params.logit_bias.empty()) {
            processors.push_back(create_processor("logit_bias"));
        }

        return processors;
    }

}
