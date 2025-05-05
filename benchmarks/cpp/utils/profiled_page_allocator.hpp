#pragma once

#include "engine/page_allocator.hpp"
#include "utils/tracy_wrapper.hpp"
#include <optional>
#include <cstdint>
#include <utility>

namespace pie_core::profiling {

template<typename AllocatorT>
class ProfiledAllocatorWrapper {
    AllocatorT& base_allocator_;
    size_t memory_per_page_ = 0;
    size_t total_memory_ = 0;

    // Private helper to calculate memory per page
    size_t calculate_memory_per_page() const {
        if (base_allocator_.size() == 0) return 0;

        // Calculate based on the first page's dimensions (all pages should be identical)
        const auto& first_page = base_allocator_.get_page(0);
        return engine::TOKEN_CAPACITY_PER_PAGE * first_page.num_heads() * first_page.head_dim() * 2 * sizeof(int8_t); // Both key and value
    }

public:
    // Constructor takes a reference to the allocator to wrap
    explicit ProfiledAllocatorWrapper(AllocatorT& allocator) : base_allocator_(allocator) {
        PIE_PROFILE_FUNCTION();
        #if defined(TRACY_ENABLE)
        memory_per_page_ = calculate_memory_per_page();
        total_memory_ = memory_per_page_ * base_allocator_.size();

        // Initial Tracy plots
        TracyPlot("PageAllocator/TotalPages", static_cast<int64_t>(base_allocator_.size()));
        TracyPlot("PageAllocator/FreePages", static_cast<int64_t>(base_allocator_.get_num_free_pages()));
        TracyPlot("PageAllocator/MemoryUtilization_Percent", 0.0);
        #endif
    }

    // Prevent copying/moving when holding a reference
    ProfiledAllocatorWrapper(const ProfiledAllocatorWrapper&) = delete;
    ProfiledAllocatorWrapper& operator=(const ProfiledAllocatorWrapper&) = delete;
    ProfiledAllocatorWrapper(ProfiledAllocatorWrapper&&) = delete;
    ProfiledAllocatorWrapper& operator=(ProfiledAllocatorWrapper&&) = delete;

    // --- Wrapped Methods ---
    std::optional<uint32_t> allocate_page() {
        PIE_PROFILE_ZONE("PageAllocator::allocate_page");
        auto result = base_allocator_.allocate_page();
        #if defined(TRACY_ENABLE)
        size_t num_free = base_allocator_.get_num_free_pages();
        TracyPlot("PageAllocator/FreePages", static_cast<int64_t>(num_free));
        TracyPlot("PageAllocator/MemoryUtilization_Percent", get_memory_utilization_percent());
        #endif
        return result;
    }

    void free_page(uint32_t page_id) {
        PIE_PROFILE_ZONE("PageAllocator::free_page");
        base_allocator_.free_page(page_id);
        #if defined(TRACY_ENABLE)
        size_t num_free = base_allocator_.get_num_free_pages();
        TracyPlot("PageAllocator/FreePages", static_cast<int64_t>(num_free));
        TracyPlot("PageAllocator/MemoryUtilization_Percent", get_memory_utilization_percent());
        #endif
    }

    void add_ref(uint32_t page_id) {
        PIE_PROFILE_ZONE("PageAllocator::add_ref");
        base_allocator_.add_ref(page_id);
    }

    engine::KVPage& get_page(uint32_t page_id) {
        // Don't profile simple getters unless needed for high-frequency calls
        return base_allocator_.get_page(page_id);
    }

    const engine::KVPage& get_page(uint32_t page_id) const {
        return base_allocator_.get_page(page_id);
    }

    [[nodiscard]] size_t size() const {
        return base_allocator_.size();
    }

    [[nodiscard]] size_t get_num_free_pages() const {
        return base_allocator_.get_num_free_pages();
    }

    // Add memory statistics getters
    [[nodiscard]] size_t get_total_memory() const {
        #if defined(TRACY_ENABLE)
        return total_memory_;
        #else
        // Calculate on-demand if not using Tracy
        return calculate_memory_per_page() * base_allocator_.size();
        #endif
    }

    [[nodiscard]] double get_memory_utilization_percent() const {
        size_t used_pages = base_allocator_.size() - base_allocator_.get_num_free_pages();
        double percentage = (base_allocator_.size() > 0)
            ? (static_cast<double>(used_pages) / base_allocator_.size()) * 100.0
            : 0.0;

        return percentage;
    }

    // Expose the underlying allocator if necessary
    AllocatorT& raw() { return base_allocator_; }
    const AllocatorT& raw() const { return base_allocator_; }
};

// ProfiledPageAllocator is a specialization of the wrapper for PageAllocator
using ProfiledPageAllocator = ProfiledAllocatorWrapper<engine::PageAllocator>;

// --- Conditional Type alias ---
#if defined(TRACY_ENABLE)
    using BenchAllocatorType = ProfiledAllocatorWrapper<engine::PageAllocator>;
#else
    using BenchAllocatorType = engine::PageAllocator; // Use the real one if Tracy is off
#endif

} // namespace pie_core::profiling
