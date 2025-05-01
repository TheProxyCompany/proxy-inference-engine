#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdint>
#include <optional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include "page.hpp"

namespace mx = mlx::core;

namespace pie_core {

    // Forward declare KVPage if needed, but including page.hpp is fine
    // struct KVPage;

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
        // Assigns the page to the given sequence_id internally.
        std::optional<uint32_t> allocate_page(uint32_t sequence_id);

        // Returns a page ID to the free list.
        void free_page(uint32_t page_id);

        // Gets a reference to the KVPage object associated with a page ID.
        // Throws if page_id is invalid.
        KVPage& get_page(uint32_t page_id);
        // Const version
        const KVPage& get_page(uint32_t page_id) const;

        // Gets the total number of pages managed by the allocator.
        [[nodiscard]] size_t size() const { return page_pool_.size(); }

        // Gets the approximate number of free pages (can be racy).
        // For monitoring/debugging, not for precise control flow.
        [[nodiscard]] size_t get_num_free_pages() const;


    private:
        // Node structure for the lock-free stack (Treiber stack)
        struct FreeNode {
            uint32_t page_index;
            FreeNode* next; // Pointer to the next free node
        };

        std::vector<KVPage> page_pool_; // Contiguous storage for all pages
        std::vector<FreeNode> node_pool_; // Pre-allocated nodes for the stack

        // Atomic head pointer for the Treiber stack
        // Points to the top FreeNode in the stack of free pages
        std::atomic<FreeNode*> head_{nullptr};

        // Atomic counter for approximate free page count (optional but useful)
        std::atomic<size_t> num_free_pages_{0};

        // Helper to push onto the lock-free stack
        void push_free_list(FreeNode* node);

        // Helper to pop from the lock-free stack
        FreeNode* pop_free_list();
    };

}
