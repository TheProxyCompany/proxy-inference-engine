#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief Configuration for Rotary Positional Embeddings (RoPE).
     */
    struct RoPEConfig {
        int dims;
        bool traditional = false;
        float base = 10000.0f;
        float scale = 1.0f;
    };

    /**
     * @brief Applies Rotary Positional Embeddings to input queries and keys.
     */
    class RoPE {
    public:
        /**
         * @brief Constructs a RoPE layer.
         * @param config Configuration parameters for RoPE.
         */
        explicit RoPE(const RoPEConfig& config);

        RoPE(const RoPE&) = delete;
        RoPE& operator=(const RoPE&) = delete;
        RoPE(RoPE&&) = default;
        RoPE& operator=(RoPE&&) = default;
        ~RoPE() = default;

        /**
         * @brief Applies RoPE to the input tensor.
         * @param x Input tensor (typically Queries or Keys).
         * @param offset Positional offset for KV cache. Default: 0.
         * @return Tensor with rotary embeddings applied.
         */
        mx::array forward(const mx::array& x, int offset = 0) const;
        mx::array operator()(const mx::array& x, int offset = 0) const { return forward(x, offset); }

    private:
        RoPEConfig config_;
    };

} // namespace pie_core::layers
