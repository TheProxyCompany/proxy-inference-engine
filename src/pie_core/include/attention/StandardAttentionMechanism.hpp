#pragma once

#include "attention/IAttentionMechanism.hpp" // Include the interface
#include <mlx/mlx.h> // For mx::array

namespace pie_core::attention {

namespace mx = mlx::core;

/**
 * @brief Implements standard scaled dot-product attention using MLX built-in operations.
 *
 * This mechanism uses mx::fast::scaled_dot_product_attention and assumes
 * the necessary causal mask is created based on the input sequence length.
 * It does not use paged KV caching.
 */
class StandardAttentionMechanism : public IAttentionMechanism {
public:
    /**
     * @brief Default constructor.
     */
    StandardAttentionMechanism() = default;

    /**
     * @brief Computes standard attention using mx::fast::scaled_dot_product_attention.
     *
     * @param queries The query tensor.
     * @param keys The key tensor for the current step.
     * @param values The value tensor for the current step.
     * @param details The batch details (used here primarily for input shape/dtype context and potentially mask creation logic).
     * @return The computed attention output tensor.
     */
    mx::array compute(
        const mx::array& queries,
        const mx::array& keys,
        const mx::array& values,
        const engine::BatchDetails& details
    ) const override;

    // Explicitly deleted copy/move operations
    StandardAttentionMechanism(const StandardAttentionMechanism&) = delete;
    StandardAttentionMechanism& operator=(const StandardAttentionMechanism&) = delete;
    StandardAttentionMechanism(StandardAttentionMechanism&&) = delete;
    StandardAttentionMechanism& operator=(StandardAttentionMechanism&&) = delete;
};

} // namespace pie_core::attention
