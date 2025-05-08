#include "attention/StandardAttentionMechanism.hpp"
#include "engine/batch_details.hpp" // Need this for details.sequence_ids.size() etc.
#include <mlx/fast.h>              // For scaled_dot_product_attention
#include <mlx/ops.h>               // For triu, full
#include <limits>                  // For infinity
#include <stdexcept>               // For runtime_error
#include <spdlog/spdlog.h>         // For logging
#include "attention/AttentionRegistry.hpp" // For auto-registration

namespace pie_core::attention {

mx::array StandardAttentionMechanism::compute(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const engine::BatchDetails& details
) const {
    spdlog::trace("StandardAttentionMechanism: Computing standard attention");

    // Input shapes are expected to be [B, H, L, D/H] after projections/RoPE in Attention layer
    if (queries.ndim() != 4 || keys.ndim() != 4 || values.ndim() != 4) {
         spdlog::error("StandardAttentionMechanism: Expected 4D input tensors [B, H, L, D/H], but got Q: {}, K: {}, V: {}",
                       queries.shape(), keys.shape(), values.shape());
         throw std::runtime_error("StandardAttentionMechanism requires 4D input tensors [B, H, L, D/H]");
    }

    int B = queries.shape(0);
    int L = queries.shape(2); // Sequence length

    // Create Causal Mask
    // This mask is applied *before* softmax inside scaled_dot_product_attention
    std::optional<mx::array> attention_mask = std::nullopt;
    if (L > 0) { // Only create mask if sequence length is positive
        mx::array mask = mx::triu(mx::full({L, L}, -std::numeric_limits<float>::infinity(), queries.dtype()), /*k=*/1);
        // The mask shape [L, L] should broadcast correctly to [B, H, L, L] within SDPA
        attention_mask = mask;
        spdlog::trace("StandardAttentionMechanism: Created causal mask of shape [{}, {}]", L, L);
    } else {
         spdlog::warn("StandardAttentionMechanism: Sequence length L is 0, skipping mask creation.");
    }


    // Perform scaled dot-product attention using MLX's optimized function
    // It handles: 1/sqrt(dk) scaling, mask application, softmax, matmul(scores, V)
    try {
        mx::array attn_output = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            /*scale=*/std::nullopt, // Uses default 1/sqrt(dk) scaling
            attention_mask          // Pass the causal mask
        );
        spdlog::trace("StandardAttentionMechanism: mx::fast::scaled_dot_product_attention completed");
        return attn_output;
    } catch (const std::exception& e) {
        spdlog::error("StandardAttentionMechanism: Error during mx::fast::scaled_dot_product_attention: {}", e.what());
        // Re-throw or handle error appropriately
        throw;
    }
}

// --- Auto-registration ---
namespace { // Use anonymous namespace
    AttentionMechanismRegistrar<StandardAttentionMechanism> registrar(engine::AttentionType::STANDARD);
} // anonymous namespace

} // namespace pie_core::attention
