#include "layers/attention.hpp"
#include "engine/batch_details.hpp"
#include <mlx/ops.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace pie_core::layers {

    Attention::Attention(const AttentionConfig& config)
        : config_(config),
          q_proj_(config.hidden_dims, config.num_heads * (config.hidden_dims / config.num_heads), config.bias),
          k_proj_(config.hidden_dims, config.num_kv_heads * (config.hidden_dims / config.num_heads), config.bias),
          v_proj_(config.hidden_dims, config.num_kv_heads * (config.hidden_dims / config.num_heads), config.bias),
          o_proj_(config.num_heads * (config.hidden_dims / config.num_heads), config.hidden_dims, config.bias),
          rope_(config.rope_config)
    {}

    // --- Forward Pass ---
    mx::array Attention::forward(
        const mx::array& hidden_state,
        const engine::BatchDetails& batch_details
    ) const {
        int B = hidden_state.shape()[0]; // Batch size
        int L = hidden_state.shape()[1]; // Sequence length of *this step*
        int head_dim = config_.hidden_dims / config_.num_heads;

        mx::array queries = q_proj_.forward(hidden_state);
        mx::array keys = k_proj_.forward(hidden_state);
        mx::array values = v_proj_.forward(hidden_state);

        queries = mx::transpose(mx::reshape(queries, {B, L, config_.num_heads, head_dim}), {0, 2, 1, 3});
        keys = mx::transpose(mx::reshape(keys, {B, L, config_.num_kv_heads, head_dim}), {0, 2, 1, 3});
        values = mx::transpose(mx::reshape(values, {B, L, config_.num_kv_heads, head_dim}), {0, 2, 1, 3});

        // 3. Apply RoPE to queries and keys for the *current* tokens
        queries = rope_.forward(queries, 0);
        keys = rope_.forward(keys, 0);

        // 4. Invoke the Paged Attention Kernel
        mx::array attn_output = invoke_paged_attention_kernel(queries, keys, values, batch_details);

        // 5. Reshape attention output back: [B, H, L, D] -> [B, L, H, D] -> [B, L, H*D]
        attn_output = mx::reshape(mx::transpose(attn_output, {0, 2, 1, 3}), {B, L, -1});

        // 6. Final output projection
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
    }

    // --- Placeholder Kernel Invocation ---
    mx::array Attention::invoke_paged_attention_kernel(
        const mx::array& queries,
        const mx::array& keys,
        const mx::array& values,
        const engine::BatchDetails& batch_details
    ) const {
        // ==============================================================
        // !!! Placeholder Implementation !!!
        // This needs to be replaced with the actual Metal kernel call
        // using the MLX C++ API for custom kernels/primitives.
        // ==============================================================
        return mx::zeros_like(queries);
    }

} // namespace pie_core::layers
