#include "logit_processors/repetition_processor.hpp"
#include "logit_processors/logit_processor_registry.hpp"
#include <mlx/ops.h>
#include <vector>
#include <cstdint>
#include <unordered_set>

namespace pie_core::logit_processors {

    mx::array RepetitionProcessor::process_logits(
        const mx::array& logits,
        const sequence::LogitsParams& params,
        const sequence::Sequence& sequence
    ) {
        // TODO: Implement repetition processor
        return logits;
    }

    namespace {
        // Register the repetition processor with the registry
        LogitProcessorRegistrar<RepetitionProcessor> registrar("repetition");
    }

}
