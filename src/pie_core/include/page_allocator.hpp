#pragma once

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "page.hpp"

namespace mx = mlx::core;

namespace pie_core {

    class PageAllocator {
        public:
            PageAllocator(size_t max_pages);
            ~PageAllocator();

            std::optional<uint32_t> allocate_page(uint32_t sequence_id);
            void free_page(uint32_t page_id);

            std::optional<KVPage&> get_page(uint32_t page_id);

        private:
            size_t max_pages;
            std::unordered_map<uint32_t, KVPage> pages;
    };

}
