#pragma once

#include "layers/linear.hpp"
#include "layers/rope.hpp"
#include "engine/batch_details.hpp"
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief Configuration struct specific to the Attention layer.
     */
    struct AttentionConfig {
        int hidden_dims;
        int num_heads;
        int num_kv_heads;
        RoPEConfig rope_config;
        bool bias = false;
    };

    /**
     * @brief Implements Multi-Head (or Grouped-Query) Attention using Paged KV Cache.
     */
    class Attention {
    public:
        /**
         * @brief Constructs the Attention layer.
         * @param config Configuration parameters.
         */
        explicit Attention(const AttentionConfig& config);

        // Rule of 5/6
        Attention(const Attention&) = delete;
        Attention& operator=(const Attention&) = delete;
        Attention(Attention&&) = default;
        Attention& operator=(Attention&&) = default;
        ~Attention() = default;

        /**
         * @brief Performs the Paged Attention forward pass.
         * @param hidden_state Input tensor from the previous layer.
         * @param batch_details Contains consolidated block tables, sequence info, etc.
         * @return Output tensor after attention calculation and output projection.
         */
        mx::array forward(
            const mx::array& hidden_state,
            const engine::BatchDetails& batch_details
        ) const;

        mx::array operator()(
            const mx::array& hidden_state,
            const engine::BatchDetails& batch_details
        ) const {
            return forward(hidden_state, batch_details);
        }

        /**
         * @brief Delegates loading weights to its internal Linear layers.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this Attention block (e.g., "self_attn.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                          const std::string& prefix);

        /**
         * @brief Delegates parameter collection to its internal Linear layers.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        AttentionConfig config_;

        // --- Sub-layers ---
        Linear q_proj_;
        Linear k_proj_;
        Linear v_proj_;
        Linear o_proj_;
        RoPE rope_;

        // --- Private Helpers ---
        /**
         * @brief Placeholder for the function that prepares inputs and calls the custom Metal kernel.
         * @param queries Queries tensor for the current step.
         * @param keys Keys tensor computed in this step.
         * @param values Values tensor computed in this step.
         * @param batch_details Contains block tables and other necessary info.
         * @return Output tensor from the attention mechanism (before o_proj).
         */
        // In attention.hpp
        mx::array invoke_paged_attention_kernel(
            const mx::array& queries,
            const mx::array& keys,
            const mx::array& values,
            const pie_core::engine::BatchDetails& batch_details
        ) const;
    };

} // namespace pie_core::layers
