#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <memory>
#include <vector>
#include "sequence/logits_params.hpp"
#include "sequence/sequence.hpp"

namespace mx = mlx::core;

namespace pie_core::logit_processors {

    // Abstract base class for all logit processors.
    class ILogitProcessor {
    public:
        virtual ~ILogitProcessor() = default;

        // Core logits processing method.
        virtual mx::array process_logits(
            const mx::array& logits,
            const sequence::LogitsParams& params,
            const sequence::Sequence& sequence
        ) = 0;
    };

} // namespace pie_core::logit_processors
