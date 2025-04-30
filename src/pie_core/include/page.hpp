#pragma once

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <mutex>
#include <string>
#include <stdexcept>


namespace mx = mlx::core;

class KVPage {
    public:
        KVPage(
            mx::array k_cache,
            mx::array v_cache,
            uint32_t page_id,
            uint32_t sequence_id,
            uint16_t num_tokens
        );

    private:
        mx::array k_cache;
        mx::array v_cache;
        uint32_t page_id;
        uint32_t sequence_id;
        uint16_t num_tokens;
};
