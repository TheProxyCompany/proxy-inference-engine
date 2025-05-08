#pragma once

#include "attention/IAttentionMechanism.hpp" // Include the interface
#include <mlx/mlx.h> // For mx::array

// Forward declare BatchDetails
namespace pie_core::engine {
    struct BatchDetails;
}

namespace pie_core::attention {

namespace mx = mlx::core;

/**
 * @brief Implements paged attention using a custom Metal kernel.
 *
 * This mechanism is responsible for interacting with the paged KV cache
 * via the consolidated block table provided in BatchDetails and invoking
 * the custom Metal kernel for computation.
 */
class PagedAttentionMechanism : public IAttentionMechanism {
public:
    /**
     * @brief Default constructor.
     * (May later take configuration specific to the kernel if needed)
     */
    PagedAttentionMechanism() = default;

    /**
     * @brief Invokes the custom paged attention Metal kernel.
     *
     * @param queries The query tensor for the current step.
     * @param keys The key tensor for the current step.
     * @param values The value tensor for the current step.
     * @param details The batch details containing sequence info, context lengths, and the consolidated block table.
     * @return The computed attention output tensor.
     */
    mx::array compute(
        const mx::array& queries,
        const mx::array& keys,
        const mx::array& values,
        const engine::BatchDetails& details
    ) const override;

    // Explicitly deleted copy/move operations
    PagedAttentionMechanism(const PagedAttentionMechanism&) = delete;
    PagedAttentionMechanism& operator=(const PagedAttentionMechanism&) = delete;
    PagedAttentionMechanism(PagedAttentionMechanism&&) = delete;
    PagedAttentionMechanism& operator=(PagedAttentionMechanism&&) = delete;

private:
    // Potential future members: kernel object handle, etc.
};

} // namespace pie_core::attention
