#include "engine/page_allocator.hpp"
#include <stdexcept>
#include <numeric>
#include <thread>

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
    // --- 0. Check if arguments are valid ---
    if (num_pages == 0) {
        throw std::invalid_argument("PageAllocator must be initialized with num_pages > 0.");
    }
    if (num_heads <= 0) {
        throw std::invalid_argument("num_heads must be positive.");
    }
    if (head_dim <= 0) {
        throw std::invalid_argument("head_dim must be positive.");
    }
    // --- 1. Initialize page_pool_ ---
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
            throw std::runtime_error(
                "Failed to construct KVPage pool: " + std::string(e.what())
            );
        }
    }
    // --- 2. Initialize node_pool_ and Build Initial Free List Stack ---
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
    }
}

// --- Public Methods Implementation ---
std::optional<uint32_t> PageAllocator::allocate_page() {
    FreeNode* node = pop_free_list();
    if (node == nullptr) {
        // Pool is exhausted
        return std::nullopt;
    }
    // fresh page
    page_pool_[node->page_index].ref_count_.store(1, std::memory_order_release);
    // caller is responsible for filling token
    page_pool_[node->page_index].set_num_tokens(0);
    return node->page_index;
}

void PageAllocator::free_page(uint32_t page_id) {
    check_page_id(page_id);
    if (page_pool_[page_id].dec_ref() == 0) {
        FreeNode* node_to_free = &node_pool_[page_id];
        push_free_list(node_to_free);
    }
}

void PageAllocator::add_ref(uint32_t page_id) {
    check_page_id(page_id);
    page_pool_[page_id].add_ref();
}

KVPage& PageAllocator::get_page(uint32_t page_id) {
    check_page_id(page_id);
    return page_pool_[page_id];
}

const KVPage& PageAllocator::get_page(uint32_t page_id) const {
    check_page_id(page_id);
    return page_pool_[page_id];
}

// -- number of free pages --
size_t PageAllocator::get_num_free_pages() const {
    return num_free_pages_.load(std::memory_order_acquire);
}

// --- Private Helper Implementation ---
void PageAllocator::check_page_id(uint32_t page_id) const {
    if (page_id >= page_pool_.size()) {
        throw std::out_of_range(
            "Page ID " + std::to_string(page_id) +
            " is out of range for pool size " + std::to_string(page_pool_.size())
        );
    }
}


// --- Treiber Stack Push ---
void PageAllocator::push_free_list(FreeNode* node) {
    // Employ the canonical Treiber stack push pattern.
    FreeNode* old_head = head_.load(std::memory_order_relaxed);
    do {
        node->next = old_head;
    } while (!head_.compare_exchange_weak(
                 old_head,
                 node,
                 std::memory_order_release,
                 std::memory_order_relaxed
             ));

    num_free_pages_.fetch_add(1, std::memory_order_relaxed);
}

// --- Treiber Stack Pop ---
PageAllocator::FreeNode* PageAllocator::pop_free_list() {
    // Spin loop for compare-and-swap
    FreeNode* current_head = head_.load(std::memory_order_acquire);
    while (current_head != nullptr && // Check if stack is empty
           !head_.compare_exchange_weak(
               current_head,
               current_head->next,
               std::memory_order_acquire,
               std::memory_order_acquire
           )) {
        // Brief pause/yield to reduce contention
        // std::this_thread::yield();
    }

    if (current_head != nullptr) {
        // Decrement free count *after* successful pop
        num_free_pages_.fetch_sub(1, std::memory_order_relaxed);
    }
    // Return the node we popped (or nullptr if stack was empty)
    return current_head;
}


} // namespace pie_core
