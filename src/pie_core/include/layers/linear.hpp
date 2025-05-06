#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief Applies a linear transformation (y = xW^T + b).
     */
    class Linear {
    public:
        /**
         * @brief Constructs a Linear layer.
         * @param input_dims Dimensionality of the input features.
         * @param output_dims Dimensionality of the output features.
         * @param bias Whether to include a bias term.
         */
        Linear(int input_dims, int output_dims, bool bias = true);

        Linear(const Linear&) = delete;
        Linear& operator=(const Linear&) = delete;
        Linear(Linear&&) = default;
        Linear& operator=(Linear&&) = default;
        ~Linear() = default;

        /**
         * @brief Performs the forward pass: x @ W.T + bias.
         * @param x Input tensor.
         * @return Output tensor.
         */
        mx::array forward(const mx::array& x) const;
        mx::array operator()(const mx::array& x) const { return forward(x); }

        /**
         * @brief Loads weights (weight, bias) from a map using a prefix.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this layer (e.g., "mlp.gate_proj.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix);

        /**
         * @brief Appends pointers to the layer's parameters (weight, bias) to the vector.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        mx::array weights_;
        std::optional<mx::array> bias_;
        bool should_bias_;
    };

} // namespace pie_core::layers
