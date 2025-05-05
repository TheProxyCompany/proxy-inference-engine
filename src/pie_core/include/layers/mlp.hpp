#pragma once

#include "layers/linear.hpp" // Depends on Linear
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief Implements the Feed-Forward Network block (MLP) typically found in Transformers.
     *        Uses SiLU activation suitable for Llama models.
     */
    class MLP {
    public:
        /**
         * @brief Constructs an MLP layer.
         * @param dim Input and output dimension.
         * @param hidden_dim Hidden dimension (intermediate size).
         */
        MLP(int dim, int hidden_dim);

        // Rule of 5/6
        MLP(const MLP&) = delete;
        MLP& operator=(const MLP&) = delete;
        MLP(MLP&&) = default;
        MLP& operator=(MLP&&) = default;
        ~MLP() = default;

        /**
         * @brief Performs the forward pass: down_proj(silu(gate_proj(x)) * up_proj(x)).
         * @param x Input tensor.
         * @return Output tensor.
         */
        mx::array forward(const mx::array& x) const;
        mx::array operator()(const mx::array& x) const { return forward(x); }

        /**
         * @brief Delegates loading weights to its internal Linear layers.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this MLP block (e.g., "mlp.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                          const std::string& prefix);

        /**
         * @brief Delegates parameter collection to its internal Linear layers.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        Linear gate_proj_;
        Linear down_proj_;
        Linear up_proj_;
    };

} // namespace pie_core::layers
