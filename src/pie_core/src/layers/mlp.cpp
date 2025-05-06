#include "layers/mlp.hpp"
#include "layers/activation_functions.hpp"

namespace pie_core::layers {

    MLP::MLP(int dim, int hidden_dim)
        : gate_proj_(dim, hidden_dim, /*bias=*/false),
          down_proj_(hidden_dim, dim, /*bias=*/false),
          up_proj_(dim, hidden_dim, /*bias=*/false)
    {}

    mx::array MLP::forward(const mx::array& x) const {
        mx::array silu_output = silu(gate_proj_.forward(x));
        mx::array up_output = up_proj_.forward(x);
        mx::array intermediate = mx::multiply(silu_output, up_output);
        return down_proj_.forward(intermediate);
    }

    void MLP::load_weights(const std::unordered_map<std::string, mx::array>& weights, const std::string& prefix) {
        std::string gate_prefix = prefix + "gate_proj.";
        std::string down_prefix = prefix + "down_proj.";
        std::string up_prefix = prefix + "up_proj.";

        gate_proj_.load_weights(weights, gate_prefix);
        down_proj_.load_weights(weights, down_prefix);
        up_proj_.load_weights(weights, up_prefix);
    }

    void MLP::collect_parameters(std::vector<mx::array*>& params) {
        gate_proj_.collect_parameters(params);
        down_proj_.collect_parameters(params);
        up_proj_.collect_parameters(params);
    }

} // namespace pie_core::layers
