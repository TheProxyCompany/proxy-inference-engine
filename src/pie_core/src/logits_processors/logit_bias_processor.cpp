#include "logit_processors/logit_bias_processor.hpp"
#include "logit_processors/logit_processor_registry.hpp"
#include <mlx/ops.h>

namespace mx = mlx::core;

namespace pie_core::logit_processors {

    mx::array LogitBiasProcessor::process_logits(
        const mx::array& logits,
        const sequence::LogitsParams& params,
        const sequence::Sequence& sequence
    ) {
        // TODO: Implement logit bias processor
        return logits;
    }

    namespace {
        // Register the logit bias processor with the registry
        LogitProcessorRegistrar<LogitBiasProcessor> registrar("logit_bias");
    }

}
