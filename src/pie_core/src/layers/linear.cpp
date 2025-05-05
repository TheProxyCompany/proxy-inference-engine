#include "layers/linear.hpp"

namespace pie_core::layers {

    Linear::Linear(int input_dims, int output_dims, bool bias)
        :
        should_bias_(bias),
        weights_(
            mx::random::uniform(
                -std::sqrt(1.0 / input_dims), // lower bound
                std::sqrt(1.0 / input_dims), // upper bound
                {output_dims, input_dims})), // shape
        bias_(
            bias ?
            std::optional{mx::zeros({output_dims})} : // if bias is true, initialize bias to zeros
            std::nullopt)
        {}

    mx::array Linear::forward(const mx::array& x) const {
        if (bias_.has_value()) {
            return mx::addmm(*bias_, x, mx::transpose(weights_));
        } else {
            return mx::matmul(x, mx::transpose(weights_));
        }
    }

    void Linear::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        std::string weight_key = prefix + "weight";
        std::string bias_key = prefix + "bias";

        try {
            weights_ = weights.at(weight_key);
            if (should_bias_ && weights.count(bias_key)) {
                bias_ = weights.at(bias_key);
            } else {
                bias_ = std::nullopt;
            }
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Error loading weights for Linear layer with prefix '" + prefix + "': " + e.what());
        }
    }

    void Linear::collect_parameters(std::vector<mx::array*>& params) {
        params.push_back(&weights_);
        if (should_bias_ && bias_.has_value()) {
            params.push_back(&bias_.value());
        }
    }

} // namespace pie_core::layers
