#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief Applies Root Mean Square Layer Normalization.
     */
    class RMSNorm {
    public:
        /**
         * @brief Constructs an RMSNorm layer.
         * @param dims The feature dimension to normalize over.
         * @param eps Epsilon for numerical stability. Default: 1e-5.
         */
        RMSNorm(int dims, float eps = 1e-5);

        // Rule of 5/6
        RMSNorm(const RMSNorm&) = delete;
        RMSNorm& operator=(const RMSNorm&) = delete;
        RMSNorm(RMSNorm&&) = default;
        RMSNorm& operator=(RMSNorm&&) = default;
        ~RMSNorm() = default;

        /**
         * @brief Performs the RMS Normalization.
         * @param x Input tensor.
         * @return Normalized tensor.
         */
        mx::array forward(const mx::array& x) const;
        mx::array operator()(const mx::array& x) const { return forward(x); }

        /**
         * @brief Loads the "weight" parameter from a map using a prefix.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this layer (e.g., "model.norm.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                          const std::string& prefix);

        /**
         * @brief Appends pointer to the layer's "weight" parameter to the vector.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        float eps_;
        mx::array weight_;
    };

} // namespace pie_core::layers
