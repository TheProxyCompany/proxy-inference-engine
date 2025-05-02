#pragma once

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <cstdint>
#include <atomic>
#include <cassert>


namespace mx = mlx::core;

namespace pie_core {

    constexpr size_t TOKEN_CAPACITY_PER_PAGE = 64;
    static_assert((TOKEN_CAPACITY_PER_PAGE & (TOKEN_CAPACITY_PER_PAGE-1)) == 0,
                  "TOKENS_PER_PAGE must be a power of two");

    alignas(64) struct KVPage {

        KVPage(
            int32_t num_heads,
            int32_t head_dim,
            int32_t page_id,
            mx::Dtype cache_dtype = mx::int8,
            mx::Dtype scale_dtype = mx::float16
        ):  num_heads_(num_heads),
            head_dim_(head_dim),
            key_cache_(mx::zeros({TOKEN_CAPACITY_PER_PAGE, num_heads, head_dim}, cache_dtype)),
            value_cache_(mx::zeros({TOKEN_CAPACITY_PER_PAGE, num_heads, head_dim}, cache_dtype)),
            key_cache_scale_(mx::ones({num_heads, 1}, scale_dtype)),
            value_cache_scale_(mx::ones({num_heads, 1}, scale_dtype)),
            page_id_(page_id),
            num_tokens_{0},
            ref_count_{0}
        {

        }

        // delete copy constructor and assignment operator
        KVPage(const KVPage&) = delete;
        KVPage& operator=(const KVPage&) = delete;

        [[nodiscard]] int32_t num_heads()   const noexcept { return num_heads_;              }
        [[nodiscard]] int32_t head_dim()    const noexcept { return head_dim_;               }
        [[nodiscard]] int32_t page_id()     const noexcept { return page_id_;                }
        [[nodiscard]] size_t num_tokens()   const noexcept { return num_tokens_;             }
        [[nodiscard]] size_t capacity()     const noexcept { return TOKEN_CAPACITY_PER_PAGE; }

        mx::array& key_cache()         noexcept { return key_cache_;        }
        mx::array& value_cache()       noexcept { return value_cache_;      }
        mx::array& key_cache_scale()   noexcept { return key_cache_scale_;  }
        mx::array& value_cache_scale() noexcept { return value_cache_scale_;}

        // Atomically increment the reference count.
        // Returns the new count.
        uint32_t add_ref() {
            #ifndef NDEBUG
            assert(ref_count_.load(std::memory_order_acquire) > 0 && "add_ref on free page");
            #endif
            return ref_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
        }

        // Atomically decrement the reference count.
        // Returns the new count.
        uint32_t dec_ref() {
            #ifndef NDEBUG
            assert(ref_count_.load(std::memory_order_acquire) > 0 && "dec_ref on free page");
            #endif
            return ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        }

        // Get the current reference count
        [[nodiscard]] uint32_t get_ref_count() const {
            return ref_count_.load(std::memory_order_acquire);
        }

        void set_num_tokens(size_t num_tokens) noexcept {
            num_tokens_.store(num_tokens, std::memory_order_release);
        }

        private:
            friend class PageAllocator;
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
            std::atomic<size_t> num_tokens_; // number of tokens in the page
            std::atomic<uint32_t> ref_count_; // Atomic counter for references

    };
}
