#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <cassert>
#include "page_allocator.hpp"

namespace mx = mlx::core;

namespace pie_core {

    class Scheduler {
    public:
        Scheduler();

        ~Scheduler() = default;

        void step();

    private:
        PageAllocator page_allocator_;
    };

}
