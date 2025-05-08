#include "layers/attention.hpp"
#include "engine/batch_details.hpp"
#include "attention/AttentionRegistry.hpp" // Include registry
#include <mlx/ops.h>
#include <cmath>
//#include <mlx/fast.h> // No longer needed here
#include <stdexcept>
#include <iostream> // Keep for potential debugging, but spdlog is better
#include <spdlog/spdlog.h> // Use spdlog

namespace pie_core::layers {

Attention::Attention(const AttentionConfig& config)
    : config_(config), // Store the config
      q_proj_(config.hidden_dims, config.num_heads * (config.hidden_dims / config.num_heads), config.bias),
      k_proj_(config.hidden_dims, config.num_kv_heads * (config.hidden_dims / config.num_heads), config.bias),
      v_proj_(config.hidden_dims, config.num_kv_heads * (config.hidden_dims / config.num_heads), config.bias),
      o_proj_(config.num_heads * (config.hidden_dims / config.num_heads), config.hidden_dims, config.bias),
      rope_(config.rope_config),
      mechanism_(nullptr) // Initialize mechanism_ to nullptr
{
    // Create the attention mechanism using the registry based on config
    spdlog::info("Attention Layer: Creating attention mechanism of type: {}", static_cast<int>(config.attention_type));
    try {
         mechanism_ = attention::AttentionRegistry::create_mechanism(config.attention_type);
         spdlog::info("Attention Layer: Successfully created attention mechanism.");
    } catch (const std::exception& e) {
         spdlog::critical("Attention Layer: Failed to create attention mechanism: {}", e.what());
         // Re-throw because the layer is unusable without a mechanism
         throw;
    }
}

// --- Forward Pass (Delegates to Mechanism) ---
mx::array Attention::forward(
    const mx::array& hidden_state,
    const engine::BatchDetails& batch_details
) const {
    if (!mechanism_) {
        throw std::runtime_error("Attention mechanism is not initialized.");
    }

    // Assume hidden_state shape [B, L, D] or [TotalTokens, D]
    // B and L determination might be complex depending on batching strategy.
    // Let's assume necessary info is in batch_details or derivable.
    int B = batch_details.sequence_ids.size(); // Example: Get B from batch_details
    int L = -1; // Need a reliable way to get L for reshape
    if (hidden_state.ndim() == 3) L = hidden_state.shape(1);
    else if (hidden_state.ndim() == 2 && B==1) L = hidden_state.shape(0);
    else { /* Need more robust L determination based on batch_details.input_lengths etc */
        spdlog::warn("Attention::forward: Cannot reliably determine L from hidden_state shape {} and B={}. Reshape might fail.", hidden_state.shape(), B);
        // As a fallback, maybe use max(input_lengths) or total_tokens? Risky.
        // Let's assume L=1 for decode steps if shape is [TotalTokens, D] ? Needs refinement.
        if(hidden_state.ndim() == 2) L = 1; // Hacky assumption for decode
        else throw std::runtime_error("Cannot determine L for QKV reshape.");
    }


    int head_dim = config_.hidden_dims / config_.num_heads;

    // 1. Project Q, K, V
    mx::array queries = q_proj_.forward(hidden_state);
    mx::array keys = k_proj_.forward(hidden_state);
    mx::array values = v_proj_.forward(hidden_state);

    // 2. Reshape and transpose for multi-head attention: -> [B, H, L, D/H]
    // This reshape assumes hidden_state was effectively [B, L, D] or equivalent.
    // If hidden_state is [TotalTokens, D], reshape needs care using batch_details.input_lengths.
    // TODO: Refine QKV reshape based on actual Scheduler packing strategy.
    queries = mx::transpose(mx::reshape(queries, {B, L, config_.num_heads, head_dim}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, config_.num_kv_heads, head_dim}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, config_.num_kv_heads, head_dim}), {0, 2, 1, 3});


    // 3. Apply RoPE based on positions in batch_details
    // RoPE expects input shape [B, H, L, D/H] and applies based on L dimension using offset.
    // The `positions` array in batch_details should contain the correct absolute positions.
    // TODO: Verify RoPE handles the positions correctly when L > 1.
    // For now, assume offset=0 for simplicity until Scheduler provides correct positions.
    int offset = 0; // Placeholder - should come from batch_details or sequence state
    queries = rope_.forward(queries, offset);
    keys = rope_.forward(keys, offset);

    // 4. Delegate attention computation to the selected mechanism
    mx::array attn_output = mechanism_->compute(queries, keys, values, batch_details);

    // 5. Reshape attention output back: [B, H, L, D/H] -> [B, L, H, D/H] -> [B, L, D] or [TotalTokens, D]
    // TODO: Reshape needs to be consistent with the input hidden_state packing.
    attn_output = mx::reshape(mx::transpose(attn_output, {0, 2, 1, 3}), {B, L, -1}); // Assumes [B, L, D] output desired

    // 6. Final output projection (linear layer)
    return o_proj_.forward(attn_output);
    }

// --- Weight Loading ---
void Attention::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
    try {
        q_proj_.load_weights(weights, prefix + "q_proj.");
        k_proj_.load_weights(weights, prefix + "k_proj.");
        v_proj_.load_weights(weights, prefix + "v_proj.");
        o_proj_.load_weights(weights, prefix + "o_proj.");
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("Error loading weights for Attention layer with prefix '" + prefix + "': " + e.what());
    }
}

// --- Parameter Collection ---
void Attention::collect_parameters(std::vector<mx::array*>& params) {
    q_proj_.collect_parameters(params);
    k_proj_.collect_parameters(params);
    v_proj_.collect_parameters(params);
    o_proj_.collect_parameters(params);
    // Note: The attention mechanism itself (IAttentionMechanism) might not have
    // directly loadable parameters in this design. If a mechanism *did* have params
    // (e.g., specific learned scaling factors), it would need its own load/collect methods.
}

} // namespace pie_core::layers
