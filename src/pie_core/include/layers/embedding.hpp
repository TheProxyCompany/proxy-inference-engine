#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mx = mlx::core;

namespace pie_core::layers {

    /**
     * @brief A simple lookup table mapping token IDs to embeddings.
     */
    class Embedding {
    public:
        /**
         * @brief Constructs an Embedding layer.
         * @param num_embeddings Vocabulary size.
         * @param dims Embedding dimensions.
         */
        Embedding(int num_embeddings, int dims);

        // Rule of 5/6
        Embedding(const Embedding&) = delete;
        Embedding& operator=(const Embedding&) = delete;
        Embedding(Embedding&&) = default;
        Embedding& operator=(Embedding&&) = default;
        ~Embedding() = default;

        /**
         * @brief Performs the embedding lookup.
         * @param x Input tensor of token IDs.
         * @return Output tensor of embeddings.
         */
        mx::array forward(const mx::array& x) const;
        mx::array operator()(const mx::array& x) const { return forward(x); }

        /**
         * @brief Use embedding weights as a linear layer (e.g., for tied output projection).
         * @param x Input tensor.
         * @return Output tensor (x @ weight.T).
         */
        mx::array as_linear(const mx::array& x) const;

        /**
         * @brief Loads the "weight" parameter from a map using a prefix.
         * @param weights Map containing all model weights.
         * @param prefix Prefix for keys belonging to this layer (e.g., "model.embed_tokens.").
         */
        void load_weights(const std::unordered_map<std::string, mx::array>& weights,
                          const std::string& prefix);

        /**
         * @brief Appends pointer to the layer's "weight" parameter to the vector.
         * @param params Vector to which parameter pointers will be added.
         */
        void collect_parameters(std::vector<mx::array*>& params);

    private:
        mx::array weights_;
        int num_embeddings_;
        int dims_;
    };

} // namespace pie_core::layers
