#include "layers/embedding.hpp"
#include <cmath>
#include <stdexcept>

namespace pie_core::layers {

    Embedding::Embedding(int num_embeddings, int dims)
        : num_embeddings_(num_embeddings),
          dims_(dims),
          weights_(mx::random::normal({num_embeddings_, dims_}, 0.0, std::sqrt(1.0 / dims_)))
    {}

    mx::array Embedding::forward(const mx::array& x) const {
        // Performs the lookup: weight_[x]
        return mx::take(weights_, x, 0);
    }

    mx::array Embedding::as_linear(const mx::array& x) const {
        // Performs the linear projection: x @ weight_.T
        return mx::matmul(x, mx::transpose(weights_));
    }

    void Embedding::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        std::string weight_key = prefix + "weight";
        try {
            if (weights.count(weight_key)) {
                 const auto& loaded_weight = weights.at(weight_key);
                 if (loaded_weight.shape(0) != num_embeddings_ || loaded_weight.shape(1) != dims_) {
                      throw std::runtime_error("Mismatched shape for embedding weight: " + weight_key);
                 }
                 weights_ = loaded_weight;
            } else {
                 throw std::out_of_range("Weight key not found: " + weight_key);
            }
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Error loading weights for Embedding layer with prefix '" + prefix + "': " + e.what());
        }
    }

    void Embedding::collect_parameters(std::vector<mx::array*>& params) {
        params.push_back(&weights_);
    }

} // namespace pie_core::layers
