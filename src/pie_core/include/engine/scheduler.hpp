#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <cassert>

#include "memory/page_allocator.hpp"
#include "sequence.hpp"

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
