#pragma once

#include "logit_processors/logit_processor.hpp"
#include "sequence/logits_params.hpp"
#include <mlx/array.h>

namespace mx = mlx::core;

namespace pie_core::logit_processors {

    class RepetitionProcessor : public ILogitProcessor {
    public:
        // Implement the overridden method
        mx::array process_logits(
            const mx::array& logits,
            const sequence::LogitsParams& params,
            const sequence::Sequence& sequence
        ) override;
    };

}
