#include "engine/page_allocator.hpp"
#include <stdexcept>
#include <numeric>
#include <thread>
#include <spdlog/spdlog.h>

namespace pie_core::engine {

PageAllocator::PageAllocator(
    size_t num_pages,
    int32_t num_heads,
    int32_t head_dim,
    mx::Dtype cache_dtype,
    mx::Dtype scale_dtype
) :
    page_pool_(),      // Initialize empty first
    node_pool_(num_pages), // Allocate space for nodes
    head_{nullptr},        // Initialize atomic head pointer
    num_free_pages_{num_pages} // Initialize atomic free count
{
    spdlog::info("PageAllocator: Initializing with num_pages={}, num_heads={}, head_dim={}",
                num_pages, num_heads, head_dim);

    // --- 0. Check if arguments are valid ---
    if (num_pages == 0) {
        spdlog::error("PageAllocator: Invalid initialization with num_pages=0");
        throw std::invalid_argument("PageAllocator must be initialized with num_pages > 0.");
    }
    if (num_heads <= 0) {
        spdlog::error("PageAllocator: Invalid initialization with num_heads={}", num_heads);
        throw std::invalid_argument("num_heads must be positive.");
    }
    if (head_dim <= 0) {
        spdlog::error("PageAllocator: Invalid initialization with head_dim={}", head_dim);
        throw std::invalid_argument("head_dim must be positive.");
    }

    // --- 1. Initialize page_pool_ ---
    spdlog::debug("PageAllocator: Reserving space for {} pages in page_pool_", num_pages);
    page_pool_.reserve(num_pages); // Reserve space to avoid reallocations

    for (size_t page_id = 0; page_id < num_pages; ++page_id) {
        try {
            // Construct KVPage in place
            page_pool_.emplace_back(
                num_heads,
                head_dim,
                static_cast<int32_t>(page_id),
                cache_dtype,
                scale_dtype
            );
        } catch (const std::exception& e) {
            // Handle potential errors during mx::array creation in KVPage constructor
            spdlog::critical("PageAllocator: Failed to construct KVPage pool at page_id={}: {}", page_id, e.what());
            throw std::runtime_error(
                "Failed to construct KVPage pool: " + std::string(e.what())
            );
        }
    }
    spdlog::debug("PageAllocator: Constructed {} pages in page_pool_", num_pages);

    // --- 2. Initialize node_pool_ and Build Initial Free List Stack ---
    spdlog::debug("PageAllocator: Building free list with {} nodes", num_pages);
    for (size_t page_id = 0; page_id < num_pages; ++page_id) {
        node_pool_[page_id].page_index = page_id;
        // Link current node to the next, unless it's the last node
        if (page_id < num_pages - 1) {
            node_pool_[page_id].next = &node_pool_[page_id + 1];
        } else {
            // Explicitly set the last node's next pointer to null
            node_pool_[page_id].next = nullptr;
        }
    }

    // --- 3. Set the atomic head_ pointer ---
    if (num_pages > 0) {
        head_.store(&node_pool_[0], std::memory_order_relaxed);
        spdlog::debug("PageAllocator: Free list head initialized to page_id=0");
    }

    spdlog::info("PageAllocator: Initialization complete. {} pages available.", num_free_pages_.load());
}

// --- Public Methods Implementation ---
std::optional<uint32_t> PageAllocator::allocate_page() {
    FreeNode* node = pop_free_list();
    if (node == nullptr) {
        // Pool is exhausted
        spdlog::warn("PageAllocator: No free pages available, allocation failed");
        return std::nullopt;
    }

    uint32_t page_id = node->page_index;
    // fresh page
    page_pool_[page_id].ref_count_.store(1, std::memory_order_release);
    // caller is responsible for filling token
    page_pool_[page_id].set_num_tokens(0);

    spdlog::debug("PageAllocator: Allocated page_id={}, remaining free pages: {}",
                 page_id, num_free_pages_.load(std::memory_order_relaxed));
    return page_id;
}

void PageAllocator::free_page(uint32_t page_id) {
    try {
        check_page_id(page_id);

        uint32_t ref_count = page_pool_[page_id].dec_ref();
        spdlog::trace("PageAllocator: Decremented reference count for page_id={}, new ref_count={}",
                     page_id, ref_count);

        if (ref_count == 0) {
            FreeNode* node_to_free = &node_pool_[page_id];
            push_free_list(node_to_free);
            spdlog::debug("PageAllocator: Freed page_id={}, returning to free list. Total free pages: {}",
                         page_id, num_free_pages_.load(std::memory_order_relaxed));
        } else {
            spdlog::trace("PageAllocator: Not returning page_id={} to free list yet, ref_count={}",
                         page_id, ref_count);
        }
    } catch (const std::exception& e) {
        spdlog::error("PageAllocator: Error while freeing page_id={}: {}", page_id, e.what());
        throw; // Re-throw the exception
    }
}

void PageAllocator::add_ref(uint32_t page_id) {
    try {
        check_page_id(page_id);
        uint32_t new_ref_count = page_pool_[page_id].add_ref();
        spdlog::trace("PageAllocator: Incremented reference count for page_id={}, new ref_count={}",
                     page_id, new_ref_count);
    } catch (const std::exception& e) {
        spdlog::error("PageAllocator: Error while adding reference to page_id={}: {}", page_id, e.what());
        throw; // Re-throw the exception
    }
}

KVPage& PageAllocator::get_page(uint32_t page_id) {
    try {
        check_page_id(page_id);
        spdlog::trace("PageAllocator: Retrieved page_id={} (non-const)", page_id);
        return page_pool_[page_id];
    } catch (const std::exception& e) {
        spdlog::error("PageAllocator: Error while retrieving page_id={}: {}", page_id, e.what());
        throw; // Re-throw the exception
    }
}

const KVPage& PageAllocator::get_page(uint32_t page_id) const {
    try {
        check_page_id(page_id);
        spdlog::trace("PageAllocator: Retrieved page_id={} (const)", page_id);
        return page_pool_[page_id];
    } catch (const std::exception& e) {
        spdlog::error("PageAllocator: Error while retrieving const page_id={}: {}", page_id, e.what());
        throw; // Re-throw the exception
    }
}

// -- number of free pages --
size_t PageAllocator::get_num_free_pages() const {
    size_t free_pages = num_free_pages_.load(std::memory_order_acquire);
    spdlog::trace("PageAllocator: Current free page count: {}", free_pages);
    return free_pages;
}

// --- Private Helper Implementation ---
void PageAllocator::check_page_id(uint32_t page_id) const {
    if (page_id >= page_pool_.size()) {
        spdlog::error("PageAllocator: Invalid page ID {} (max valid ID: {})",
                     page_id, page_pool_.size() - 1);
        throw std::out_of_range(
            "Page ID " + std::to_string(page_id) +
            " is out of range for pool size " + std::to_string(page_pool_.size())
        );
    }
}


// --- Treiber Stack Push ---
void PageAllocator::push_free_list(FreeNode* node) {
    uint32_t page_id = node->page_index;
    spdlog::trace("PageAllocator: Pushing page_id={} to free list", page_id);

    // Employ the canonical Treiber stack push pattern.
    FreeNode* old_head = head_.load(std::memory_order_relaxed);
    int attempt_count = 0;
    do {
        node->next = old_head;
        attempt_count++;
        if (attempt_count > 100) {
            spdlog::warn("PageAllocator: High contention on free list push for page_id={}, attempt #{}",
                        page_id, attempt_count);
        }
    } while (!head_.compare_exchange_weak(
                 old_head,
                 node,
                 std::memory_order_release,
                 std::memory_order_relaxed
             ));

    num_free_pages_.fetch_add(1, std::memory_order_relaxed);
    spdlog::trace("PageAllocator: Successfully pushed page_id={} to free list after {} attempts, free pages: {}",
                 page_id, attempt_count, num_free_pages_.load(std::memory_order_relaxed));
}

// --- Treiber Stack Pop ---
PageAllocator::FreeNode* PageAllocator::pop_free_list() {
    spdlog::trace("PageAllocator: Attempting to pop a page from free list");

    // Spin loop for compare-and-swap
    FreeNode* current_head = head_.load(std::memory_order_acquire);
    int attempt_count = 0;

    while (current_head != nullptr && // Check if stack is empty
           !head_.compare_exchange_weak(
               current_head,
               current_head->next,
               std::memory_order_acquire,
               std::memory_order_acquire
           )) {
        // Brief pause/yield to reduce contention
        // std::this_thread::yield();
        attempt_count++;
        if (attempt_count > 100) {
            spdlog::warn("PageAllocator: High contention on free list pop, attempt #{}", attempt_count);
        }
    }

    if (current_head != nullptr) {
        // Decrement free count *after* successful pop
        num_free_pages_.fetch_sub(1, std::memory_order_relaxed);
        spdlog::trace("PageAllocator: Successfully popped page_id={} from free list after {} attempts, remaining free pages: {}",
                     current_head->page_index, attempt_count, num_free_pages_.load(std::memory_order_relaxed));
    } else {
        spdlog::debug("PageAllocator: Failed to pop from free list - no free pages available");
    }

    // Return the node we popped (or nullptr if stack was empty)
    return current_head;
}


} // namespace pie_core
