#pragma once

#include <mlx/mlx.h> // For mx::array
#include <optional>  // For std::optional potentially needed by implementations later
#include <memory>    // For std::unique_ptr

// Forward declaration to avoid circular dependency if BatchDetails needed more detail
namespace pie_core::engine {
    struct BatchDetails;
}

namespace pie_core::attention {

namespace mx = mlx::core;

/**
 * @brief Abstract interface for different attention mechanism implementations.
 *
 * This allows swapping between standard attention, paged attention, etc.,
 * without changing the core Attention layer logic.
 */
class IAttentionMechanism {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~IAttentionMechanism() = default;

    /**
     * @brief Performs the core attention computation.
     *
     * @param queries The query tensor, typically shaped [B, H, L, D/H] or [TotalTokens, D].
     * @param keys The key tensor for the current step, typically shaped [B, H, L, D/H] or [TotalTokens, D].
     * @param values The value tensor for the current step, typically shaped [B, H, L, D/H] or [TotalTokens, D].
     * @param details The batch details containing sequence info, context lengths, block tables (for paged), and attention type.
     * @return The computed attention output tensor, typically shaped [B, H, L, D/H] or [TotalTokens, D].
     */
    virtual mx::array compute(
        const mx::array& queries,
        const mx::array& keys,
        const mx::array& values,
        const engine::BatchDetails& details
    ) const = 0;

    // Prevent copying/moving of the interface itself
    IAttentionMechanism(const IAttentionMechanism&) = delete;
    IAttentionMechanism& operator=(const IAttentionMechanism&) = delete;
    IAttentionMechanism(IAttentionMechanism&&) = delete;
    IAttentionMechanism& operator=(IAttentionMechanism&&) = delete;

protected:
    // Protected constructor to allow derivation but prevent direct instantiation
    IAttentionMechanism() = default;
};

} // namespace pie_core::attention
