#include "layers/transformer_block.hpp"
#include "engine/batch_details.hpp"
#include <stdexcept>

namespace pie_core::layers {

    TransformerBlock::TransformerBlock(const TransformerBlockConfig& config )
        : input_layernorm_(config.hidden_dims, config.norm_eps),
          self_attn_(config.attn_config),
          post_attention_layernorm_(config.hidden_dims, config.norm_eps),
          mlp_(config.hidden_dims, config.mlp_hidden_dims)
    {}

    mx::array TransformerBlock::forward(
        const mx::array& hidden_state,
        const pie_core::engine::BatchDetails& batch_details
    ) const {

        // 1. Input Normalization + Residual
        mx::array residual = hidden_state;
        mx::array attn_input = input_layernorm_.forward(hidden_state);

        // 2. Self-Attention
        mx::array attn_output = self_attn_.forward(attn_input, batch_details);

        // 3. First Residual Connection
        mx::array post_attn_hidden_state = residual + attn_output;

        // 4. Post-Attention Normalization + Residual
        residual = post_attn_hidden_state;
        mx::array mlp_input = post_attention_layernorm_.forward(post_attn_hidden_state);

        // 5. MLP
        mx::array mlp_output = mlp_.forward(mlp_input);

        // 6. Second Residual Connection
        mx::array final_output = residual + mlp_output;

        return final_output;
    }

    // Load weights by delegating to sub-layers
    void TransformerBlock::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        // Delegate loading to each sub-layer with its specific prefix part
        input_layernorm_.load_weights(weights, prefix + "input_layernorm.");
        self_attn_.load_weights(weights, prefix + "self_attn.");
        post_attention_layernorm_.load_weights(weights, prefix + "post_attention_layernorm.");
        mlp_.load_weights(weights, prefix + "mlp.");
    }

    // Collect parameters by delegating to sub-layers
    void TransformerBlock::collect_parameters(std::vector<mx::array*>& params) {
        input_layernorm_.collect_parameters(params);
        self_attn_.collect_parameters(params);
        post_attention_layernorm_.collect_parameters(params);
        mlp_.collect_parameters(params);
    }

} // namespace pie_core::layers
