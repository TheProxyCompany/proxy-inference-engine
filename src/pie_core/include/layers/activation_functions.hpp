#pragma once

#include <mlx/array.h>
#include <mlx/ops.h>
#include <cmath>

namespace mx = mlx::core;

namespace pie_core::layers {

    mx::array gelu(const mx::array& x)
    {
        return x * (1 + mx::erf(x / std::sqrt(2.0))) / 2.0;
    }

    mx::array silu(const mx::array& x)
    {
        return x * mx::sigmoid(x);
    }

} // namespace pie_core::layers
