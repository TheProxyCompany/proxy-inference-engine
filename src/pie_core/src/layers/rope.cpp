#include "layers/rope.hpp"
#include <mlx/fast.h>

namespace pie_core::layers {

    RoPE::RoPE(const RoPEConfig& config)
        : config_(config)
    {}

    mx::array RoPE::forward(const mx::array& x, int offset) const {
        return mx::fast::rope(
            x,
            config_.dims,
            config_.traditional,
            config_.base,
            config_.scale,
            offset
        );
    }

} // namespace pie_core::layers
