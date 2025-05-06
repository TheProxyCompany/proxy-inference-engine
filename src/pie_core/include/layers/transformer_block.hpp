#pragma once

#include "engine/batch_details.hpp"
#include "layers/attention.hpp"
#include "layers/mlp.hpp"
#include "layers/norm.hpp"
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declare BatchDetails
namespace pie_core { struct BatchDetails; }

namespace pie_core::layers {

    struct TransformerBlockConfig {
        int hidden_dims;       // For RMSNorm, MLP
        int mlp_hidden_dims;   // For MLP
        float norm_eps;        // For RMSNorm
        AttentionConfig attn_config; // For Attention sub-layer
    };

    /**
     * @brief Represents a single block within a Transformer architecture (Attention + MLP).
     */
    class TransformerBlock {
    public:
        /**
         * @brief Constructs a TransformerBlock.
         * @param config Configuration for the TransformerBlock.
         */
        explicit TransformerBlock(const TransformerBlockConfig& config);

        // Rule of 5/6
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;
        TransformerBlock(TransformerBlock&&) = default;
        TransformerBlock& operator=(TransformerBlock&&) = default;
        ~TransformerBlock() = default;

        /**
         * @brief Performs the forward pass through the transformer block.
         * @param hidden_state Input tensor from the previous block or embedding layer.
         * @param batch_details Contains necessary info passed down to the Attention layer.
         * @return Output tensor from the block.
         */
        mx::array forward(const mx::array& hidden_state, const pie_core::engine::BatchDetails& batch_details) const;
        mx::array operator()(const mx::array& hidden_state, const pie_core::engine::BatchDetails& batch_details) const {
            return forward(hidden_state, batch_details);
        }

        /**
         * @brief Delegates loading weights to its internal Attention, MLP, and Norm layers.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this block (e.g., "model.layers.0.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                          const std::string& prefix);

        /**
         * @brief Delegates parameter collection to its internal Attention, MLP, and Norm layers.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        // Explicit member layers
        RMSNorm input_layernorm_;
        Attention self_attn_;
        RMSNorm post_attention_layernorm_;
        MLP mlp_;
    };

} // namespace pie_core::layers
