#pragma once

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

    /**
     * @brief Represents a single block within a Transformer architecture (Attention + MLP).
     */
    class TransformerBlock {
    public:
        /**
         * @brief Constructs a TransformerBlock.
         * @param config // Pass necessary parameters directly or via a struct like LlamaConfig
         *        e.g., int hidden_dims, int num_heads, int num_kv_heads, int mlp_hidden_dims,
         *              const RoPEConfig& rope_config, float norm_eps
         */
        explicit TransformerBlock(/* config parameters */);

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
        mx::array forward(const mx::array& hidden_state, const pie_core::BatchDetails& batch_details) const;
        mx::array operator()(const mx::array& hidden_state, const pie_core::BatchDetails& batch_details) const {
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
        RMSNorm input_layernorm_; // Renamed from attention_norm for clarity
        Attention self_attn_;
        RMSNorm post_attention_layernorm_; // Renamed from mlp_norm for clarity
        MLP mlp_;
    };

} // namespace pie_core::layers
