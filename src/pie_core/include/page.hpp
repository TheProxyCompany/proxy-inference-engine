#pragma once

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <cstdint>


namespace mx = mlx::core;

namespace pie_core {

    constexpr uint16_t TOKENS_PER_PAGE = 64;
    static_assert((TOKENS_PER_PAGE & (TOKENS_PER_PAGE-1)) == 0,
                  "TOKENS_PER_PAGE must be a power of two");

    struct KVPage {

        KVPage(
            int32_t num_heads,
            int32_t head_dim,
            int32_t page_id,
            mx::Dtype cache_dtype = mx::int8,
            mx::Dtype scale_dtype = mx::float16
        ):  num_heads_(num_heads),
            head_dim_(head_dim),
            key_cache_(mx::zeros({TOKENS_PER_PAGE, num_heads, head_dim}, cache_dtype)),
            value_cache_(mx::zeros_like(key_cache_)),
            key_cache_scale_(mx::ones({num_heads, 1}, scale_dtype)),
            value_cache_scale_(mx::ones_like(key_cache_scale_)),
            page_id_(page_id)
        {

        }

        [[nodiscard]] int32_t num_heads()   const { return num_heads_;    }
        [[nodiscard]] int32_t head_dim()    const { return head_dim_;      }
        [[nodiscard]] int32_t page_id()     const { return page_id_;    }
        [[nodiscard]] int32_t sequence_id() const { return sequence_id_; }
        [[nodiscard]] uint16_t num_tokens()  const { return num_tokens_;  }

        mx::array& key_cache() { return key_cache_; }
        mx::array& value_cache() { return value_cache_; }
        mx::array& key_cache_scale() { return key_cache_scale_; }
        mx::array& value_cache_scale() { return value_cache_scale_; }

        void reset(uint32_t new_seq_id) {
            sequence_id_ = new_seq_id;
            num_tokens_  = 0;
        }

        private:
            int32_t num_heads_; // number of attention heads
            int32_t head_dim_; // dimension of each attention head

            mx::array key_cache_; // [num_tokens, num_heads, head_dim]
            mx::array value_cache_; // [num_tokens, num_heads, head_dim]

            // head-wise quant = [num_heads, 1]
            // channel-wise quant = [num_heads, head_dim]
            // head-wise quant for now - TODO: test channel-wise quant

            mx::array key_cache_scale_; // [num_heads, 1]
            mx::array value_cache_scale_; // [num_heads, 1]

            int32_t page_id_ = INT32_MAX; // unique identifier for the page
            int32_t sequence_id_ = INT32_MAX; // unique identifier for the sequence
            uint16_t num_tokens_ = 0; // number of tokens in the page

    };
}
