#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <cassert>
#include "page.hpp"

namespace mx = mlx::core;

namespace pie_core::engine {

    class PageAllocator {
    public:
        // Constructor: Initializes the page pool and the free list
        PageAllocator(
            size_t num_pages,          // Total number of pages to allocate
            int32_t num_heads,         // Needed to construct KVPage
            int32_t head_dim,          // Needed to construct KVPage
            mx::Dtype cache_dtype = mx::int8,    // Passed to KVPage
            mx::Dtype scale_dtype = mx::float16  // Passed to KVPage
        );

        // Destructor (default might be okay if using std::vector and atomics)
        ~PageAllocator() = default;

        // Allocates a page ID from the free list.
        // Returns std::nullopt if the pool is exhausted.
        std::optional<uint32_t> allocate_page();

        // Decrements the reference count of the page.
        // If the count reaches 0, adds the page back to the free list.
        void free_page(uint32_t page_id);

        // Explicitly increments the reference count for a page (for sharing).
        // Use with caution - ensure the page is not already free.
        void add_ref(uint32_t page_id);

        // Gets a reference to the KVPage object associated with a page ID.
        // Throws std::out_of_range if page_id is invalid.
        KVPage& get_page(uint32_t page_id);
        const KVPage& get_page(uint32_t page_id) const;

        // Returns the number of pages in the pool.
        [[nodiscard]] size_t size() const { return page_pool_.size(); }

        // Returns the number of free pages in the pool.
        [[nodiscard]] size_t get_num_free_pages() const;

    private:
        struct FreeNode {
            uint32_t page_index;
            FreeNode* next; // Pointer to the next free node
        };

        std::vector<KVPage> page_pool_; // Owns the pages
        std::vector<FreeNode> node_pool_; // Nodes for the free list stack

        std::atomic<FreeNode*> head_{nullptr}; // Head of the free list stack
        std::atomic<size_t> num_free_pages_{0}; // Approximate count

        // Helper to validate page ID
        void check_page_id(uint32_t page_id) const;

        // Treiber stack operations - implementation in .cpp file
        void push_free_list(FreeNode* node);
        FreeNode* pop_free_list();
    };

}
