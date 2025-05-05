#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <memory>
#include "sequence/logits_params.hpp"

namespace pie_core { class Sequence; }

namespace mx = mlx::core;

namespace pie_core {

    // Abstract base class for all logit processors.
    class ILogitProcessor {
    public:
        virtual ~ILogitProcessor() = default;

        // Core sampling method.
        virtual mx::array process_logits(
            const mx::array& logits,
            const LogitsParams& params,
            const Sequence& sequence
        ) = 0;
    };

} // namespace pie_core
