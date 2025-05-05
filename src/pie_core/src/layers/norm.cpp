#include "layers/norm.hpp"
#include <mlx/fast.h>


namespace pie_core::layers {

    RMSNorm::RMSNorm(int dims, float eps)
        : eps_(eps),
          weights_(mx::ones({dims}))
    {}

    mx::array RMSNorm::forward(const mx::array& x) const {
        return mx::fast::rms_norm(x, weights_, eps_);
    }

    void RMSNorm::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        std::string weight_key = prefix + "weight";
        if (weights.count(weight_key)) {
            weights_ = weights.at(weight_key);
        }
    }

    void RMSNorm::collect_parameters(std::vector<mx::array*>& params) {
        params.push_back(&weights_);
    }

    LayerNorm::LayerNorm(int dims, float eps, bool bias)
        : eps_(eps),
          should_bias_(bias),
          weights_(mx::ones({dims})),
          bias_(bias ? std::optional{mx::zeros({dims})} : std::nullopt)
    {}

    mx::array LayerNorm::forward(const mx::array& x) const {
        return mx::fast::layer_norm(x, weights_, bias_, eps_);
    }

    void LayerNorm::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        std::string weight_key = prefix + "weight";
        std::string bias_key = prefix + "bias";

        if (weights.count(weight_key)) {
            weights_ = weights.at(weight_key);
        }
        if (should_bias_ && weights.count(bias_key)) {
            bias_ = weights.at(bias_key);
        } else {
            bias_ = std::nullopt;
        }
    }

    void LayerNorm::collect_parameters(std::vector<mx::array*>& params) {
        params.push_back(&weights_);
        if (should_bias_ && bias_.has_value()) {
            params.push_back(&bias_.value());
        }
    }

} // namespace pie_core::layers
