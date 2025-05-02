#include <gtest/gtest.h>
#include "page_allocator.hpp"
#include <vector>
#include <numeric>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>
#include <algorithm>
#include <iostream> // For potential debug/stats output

using namespace pie_core;

// --- Test Fixture ---
class PageAllocatorTest : public ::testing::Test {
protected:
    // Common constants for tests
    static constexpr int32_t DEFAULT_NUM_HEADS = 4;
    static constexpr int32_t DEFAULT_HEAD_DIM = 16;
    static constexpr size_t LARGE_POOL_SIZE = 1024; // For concurrency tests
    static constexpr size_t SMALL_POOL_SIZE = 10;   // For LIFO test
    static constexpr size_t TINY_POOL_SIZE = 4;     // For basic tests
    static constexpr size_t SINGLE_PAGE_POOL = 1;   // For edge cases
};

// --- Constructor Tests ---

TEST_F(PageAllocatorTest, ConstructorValidArgs) {
    EXPECT_NO_THROW(PageAllocator(TINY_POOL_SIZE, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM));
}

TEST_F(PageAllocatorTest, ConstructorInvalidArgs) {
    EXPECT_THROW(PageAllocator(0, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM), std::invalid_argument);
    EXPECT_THROW(PageAllocator(TINY_POOL_SIZE, 0, DEFAULT_HEAD_DIM), std::invalid_argument);
    EXPECT_THROW(PageAllocator(TINY_POOL_SIZE, DEFAULT_NUM_HEADS, 0), std::invalid_argument);
}

// --- Basic Allocation & Freeing ---

TEST_F(PageAllocatorTest, BasicExhaustAndRefill) {
    PageAllocator alloc(TINY_POOL_SIZE, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.size(), TINY_POOL_SIZE);
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE);

    std::vector<uint32_t> ids;
    ids.reserve(TINY_POOL_SIZE);
    for (size_t i = 0; i < TINY_POOL_SIZE; ++i) {
        auto id_opt = alloc.allocate_page();
        ASSERT_TRUE(id_opt.has_value()) << "Allocation failed at iteration " << i;
        ids.push_back(*id_opt);
        EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE - (i + 1));
        EXPECT_EQ(alloc.get_page(*id_opt).get_ref_count(), 1); // Initial ref count check
    }

    // Check exhaustion
    EXPECT_FALSE(alloc.allocate_page().has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), 0);

    // Free pages
    for (uint32_t id : ids) {
        alloc.free_page(id);
    }
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE); // Check if fully refilled

    // Allocate again
    auto id_opt_after = alloc.allocate_page();
    ASSERT_TRUE(id_opt_after.has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE - 1);
}

TEST_F(PageAllocatorTest, AllocationExhaustionReturnsNullopt) {
    PageAllocator alloc(2, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    ASSERT_TRUE(alloc.allocate_page().has_value());
    ASSERT_TRUE(alloc.allocate_page().has_value());
    EXPECT_FALSE(alloc.allocate_page().has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}

// --- Edge Cases ---

TEST_F(PageAllocatorTest, EdgeCaseSizeOne) {
    PageAllocator alloc(SINGLE_PAGE_POOL, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.size(), SINGLE_PAGE_POOL);
    EXPECT_EQ(alloc.get_num_free_pages(), SINGLE_PAGE_POOL);

    auto id1_opt = alloc.allocate_page();
    ASSERT_TRUE(id1_opt.has_value());
    EXPECT_EQ(*id1_opt, 0);
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_FALSE(alloc.allocate_page().has_value()); // Exhausted

    alloc.free_page(*id1_opt);
    EXPECT_EQ(alloc.get_num_free_pages(), SINGLE_PAGE_POOL);

    auto id2_opt = alloc.allocate_page();
    ASSERT_TRUE(id2_opt.has_value());
    EXPECT_EQ(*id2_opt, 0); // Should get the same page back
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}

// --- LIFO (Stack Behavior) Test ---

TEST_F(PageAllocatorTest, LIFOCheck) {
    PageAllocator alloc(SMALL_POOL_SIZE, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    std::vector<uint32_t> allocated_ids;
    allocated_ids.reserve(SMALL_POOL_SIZE);
    for (size_t i = 0; i < SMALL_POOL_SIZE; ++i) {
         auto id_opt = alloc.allocate_page();
         ASSERT_TRUE(id_opt.has_value());
         allocated_ids.push_back(*id_opt);
    }
    EXPECT_EQ(alloc.get_num_free_pages(), 0);

    // Free in reverse order of allocation
    std::vector<uint32_t> expected_realloc_order = allocated_ids; // Keep original order for comparison
    std::vector<uint32_t> freed_ids_order = allocated_ids;
    std::reverse(freed_ids_order.begin(), freed_ids_order.end());
    for (uint32_t id : freed_ids_order) {
        alloc.free_page(id);
    }
    EXPECT_EQ(alloc.get_num_free_pages(), SMALL_POOL_SIZE);

    // Allocate again and check if they come out in the order they were freed (LIFO)
    std::vector<uint32_t> reallocated_ids;
    reallocated_ids.reserve(SMALL_POOL_SIZE);
    for (size_t i = 0; i < SMALL_POOL_SIZE; ++i) {
         auto id_opt = alloc.allocate_page();
         ASSERT_TRUE(id_opt.has_value());
         reallocated_ids.push_back(*id_opt);
    }
    // Because freed order was reversed(allocated), reallocated should match original allocated
    EXPECT_EQ(reallocated_ids, expected_realloc_order);
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}

// --- Reference Counting Tests ---

TEST_F(PageAllocatorTest, SingleThreadRefCounting) {
    PageAllocator alloc(SINGLE_PAGE_POOL, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);

    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t page_id = *id_opt;

    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    ASSERT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 2);
    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 3);

    alloc.free_page(page_id); // Ref count becomes 2
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 2);

    alloc.free_page(page_id); // Ref count becomes 1
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    alloc.free_page(page_id); // Ref count becomes 0, page freed
    EXPECT_EQ(alloc.get_num_free_pages(), SINGLE_PAGE_POOL);
}

TEST_F(PageAllocatorTest, ExplicitAddRef) {
    PageAllocator alloc(SINGLE_PAGE_POOL, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t page_id = *id_opt;
    ASSERT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 2);

    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 3);

    // Test exception case with invalid page_id
    EXPECT_THROW(alloc.add_ref(999), std::out_of_range);
}

// --- Getter Tests ---

TEST_F(PageAllocatorTest, GetPageNonConst) {
    PageAllocator alloc(2, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    auto id1_opt = alloc.allocate_page();
    auto id2_opt = alloc.allocate_page();
    ASSERT_TRUE(id1_opt.has_value());
    ASSERT_TRUE(id2_opt.has_value());
    uint32_t page_id1 = *id1_opt;
    uint32_t page_id2 = *id2_opt;

    KVPage& page1 = alloc.get_page(page_id1);
    EXPECT_EQ(page1.page_id(), page_id1);
    KVPage& page2 = alloc.get_page(page_id2);
    EXPECT_EQ(page2.page_id(), page_id2);

    // Verify non-const access by calling a non-const method
    EXPECT_NO_THROW(page1.key_cache());

    EXPECT_THROW(alloc.get_page(999), std::out_of_range);
}

TEST_F(PageAllocatorTest, GetPageConst) {
    PageAllocator alloc(2, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    auto id1_opt = alloc.allocate_page();
    auto id2_opt = alloc.allocate_page();
    ASSERT_TRUE(id1_opt.has_value());
    ASSERT_TRUE(id2_opt.has_value());
    uint32_t page_id1 = *id1_opt;
    uint32_t page_id2 = *id2_opt;

    const PageAllocator& const_alloc = alloc;

    const KVPage& const_page1 = const_alloc.get_page(page_id1);
    EXPECT_EQ(const_page1.page_id(), page_id1);
    EXPECT_EQ(const_page1.get_ref_count(), 1); // Can call const methods

    const KVPage& const_page2 = const_alloc.get_page(page_id2);
    EXPECT_EQ(const_page2.page_id(), page_id2);
    EXPECT_EQ(const_page2.get_ref_count(), 1);

    EXPECT_THROW(const_alloc.get_page(999), std::out_of_range);
}

// --- Error Handling and Boundary Tests ---

TEST_F(PageAllocatorTest, InvalidIdThrows) {
    PageAllocator alloc(5, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    const uint32_t invalid_id = 5; // Boundary ID
    const uint32_t far_invalid_id = 100;

    EXPECT_THROW(alloc.get_page(invalid_id), std::out_of_range);
    EXPECT_THROW(alloc.get_page(far_invalid_id), std::out_of_range);
    EXPECT_THROW(alloc.free_page(invalid_id), std::out_of_range);
    EXPECT_THROW(alloc.add_ref(invalid_id), std::out_of_range);

    // Allocate one page to test operations on a valid ID
    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t valid_id = *id_opt;
    EXPECT_NO_THROW(alloc.get_page(valid_id));
    EXPECT_NO_THROW(alloc.add_ref(valid_id)); // Ref count now 2
    EXPECT_NO_THROW(alloc.free_page(valid_id)); // Ref count now 1
    EXPECT_NO_THROW(alloc.free_page(valid_id)); // Ref count now 0, page freed
}

TEST_F(PageAllocatorTest, CheckPageIdBoundary) {
    PageAllocator alloc(5, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    const size_t boundary_id = alloc.size(); // The first invalid ID

    EXPECT_THROW(alloc.get_page(boundary_id), std::out_of_range);
    EXPECT_THROW(alloc.free_page(boundary_id), std::out_of_range);
    EXPECT_THROW(alloc.add_ref(boundary_id), std::out_of_range);

    const PageAllocator& const_alloc = alloc;
    EXPECT_THROW(const_alloc.get_page(boundary_id), std::out_of_range);
}

// --- Multi-threading Tests ---

// Test concurrent allocations and frees. Producers free pages, one consumer allocates them.
TEST_F(PageAllocatorTest, ConcurrentAllocFreeProducersConsumer) {
    const size_t num_threads = std::min(4u, std::thread::hardware_concurrency()); // Keep thread count reasonable
    const size_t num_producers = num_threads > 1 ? num_threads - 1 : 1;
    const size_t pages_per_producer = LARGE_POOL_SIZE / num_producers;
    const size_t total_pages_to_process = pages_per_producer * num_producers; // Ensure divisibility

    PageAllocator alloc(total_pages_to_process, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.get_num_free_pages(), total_pages_to_process);

    // Pre-allocate pages to be freed by producers
    std::vector<std::vector<uint32_t>> producer_initial_pages(num_producers);
    for(size_t i = 0; i < num_producers; ++i) {
        producer_initial_pages[i].reserve(pages_per_producer);
        for(size_t j = 0; j < pages_per_producer; ++j) {
            auto id_opt = alloc.allocate_page();
            ASSERT_TRUE(id_opt.has_value()) << "Pre-allocation failed for producer " << i << " page " << j;
            producer_initial_pages[i].push_back(*id_opt);
        }
    }
    ASSERT_EQ(alloc.get_num_free_pages(), 0) << "Pool should be empty after pre-allocation";

    std::vector<std::thread> threads;
    std::atomic<size_t> consumer_alloc_count{0};
    std::atomic<bool> start_signal{false};
    std::vector<std::optional<uint32_t>> consumer_allocations(total_pages_to_process); // Store results
    std::mutex consumer_alloc_mutex; // Protect vector access if needed, though fetch_add index works here

    // Consumer thread
    threads.emplace_back([&]() {
        while(!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        for(size_t i = 0; i < total_pages_to_process; ++i) {
            std::optional<uint32_t> id_opt;
            // Spin briefly if allocator is temporarily empty
            int retries = 0;
            constexpr int max_retries = 5000; // Increased retries
            do {
                id_opt = alloc.allocate_page();
                if (!id_opt && retries < max_retries) {
                     std::this_thread::yield(); // Give producers time
                     retries++;
                } else {
                     break; // Got a page or exceeded retries
                }
            } while(true);

            ASSERT_TRUE(id_opt.has_value()) << "Consumer failed to allocate page " << i << " after retries";
            size_t index = consumer_alloc_count.fetch_add(1, std::memory_order_relaxed);
            consumer_allocations[index] = id_opt;
        }
    });

    // Producer threads
    for (size_t i = 0; i < num_producers; ++i) {
        threads.emplace_back([&, i]() {
            while(!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (uint32_t page_id : producer_initial_pages[i]) {
                alloc.free_page(page_id);
                 // Optional yield to increase contention potential
                 if (page_id % 10 == 0) std::this_thread::yield();
            }
        });
    }

    start_signal.store(true, std::memory_order_release); // Signal threads to start

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(consumer_alloc_count.load(), total_pages_to_process) << "Consumer did not allocate all expected pages";
    EXPECT_EQ(alloc.get_num_free_pages(), 0) << "Allocator should be empty after consumer finishes";

    // Verify all allocated IDs are unique
    std::set<uint32_t> unique_ids;
    size_t valid_allocs = 0;
    for(const auto& id_opt : consumer_allocations) {
        if (id_opt) {
            EXPECT_TRUE(unique_ids.insert(*id_opt).second) << "Duplicate page ID allocated: " << *id_opt;
            valid_allocs++;
        }
    }
    EXPECT_EQ(valid_allocs, total_pages_to_process) << "Number of valid allocations mismatch";
    EXPECT_EQ(unique_ids.size(), total_pages_to_process) << "Number of unique allocated IDs mismatch";
}


// Test concurrent free operations on a single, shared page
TEST_F(PageAllocatorTest, ConcurrentFreeSharedPage) {
    const int num_refs = 10; // Increased refs for more contention
    PageAllocator alloc(SINGLE_PAGE_POOL, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);

    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t page_id = *id_opt;
    ASSERT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    // Add extra references atomically
    for (int i = 1; i < num_refs; ++i) {
        alloc.add_ref(page_id);
    }
    ASSERT_EQ(alloc.get_page(page_id).get_ref_count(), num_refs);
    ASSERT_EQ(alloc.get_num_free_pages(), 0);

    std::vector<std::thread> threads;
    std::atomic<bool> start_signal{false};
    threads.reserve(num_refs);
    for(int i = 0; i < num_refs; ++i) {
        threads.emplace_back([&]() {
            while(!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            alloc.free_page(page_id); // Each thread frees one reference
        });
    }

    start_signal.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }

    // Verify page is now free (ref count should have reached 0)
    EXPECT_EQ(alloc.get_num_free_pages(), SINGLE_PAGE_POOL);

    // Verify it can be allocated again and ref count is 1
    auto id_opt_after = alloc.allocate_page();
    ASSERT_TRUE(id_opt_after.has_value());
    EXPECT_EQ(*id_opt_after, page_id); // Should get the same page back
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 1);
}

// High contention test focused on the push/pop mechanism of the free list.
// Threads repeatedly free and allocate pages from a shared pool.
TEST_F(PageAllocatorTest, HighContentionPushPopStress) {
    const size_t num_pages = 128; // Medium pool size
    const size_t num_threads = std::min(8u, std::thread::hardware_concurrency()); // Use more threads
    const size_t ops_per_thread = 2000; // Increased operations per thread

    PageAllocator alloc(num_pages, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages);

    // Each thread manages a small list of pages it owns and cycles through
    const size_t pages_per_thread_list = 4; // Small list to increase contention on pool
    std::vector<std::vector<uint32_t>> thread_local_pages(num_threads);

    // Initial allocation phase: Distribute pages among threads
    for(size_t i=0; i < num_threads; ++i) {
        thread_local_pages[i].reserve(pages_per_thread_list);
        for(size_t j=0; j < pages_per_thread_list; ++j) {
            auto id_opt = alloc.allocate_page();
            // Handle potential initial exhaustion if num_pages is small
            if (!id_opt) {
                 std::cerr << "Warning: Pool exhausted during initial allocation for HighContentionPushPopStress test." << std::endl;
                 break;
            }
            thread_local_pages[i].push_back(*id_opt);
        }
        if (thread_local_pages[i].empty()) {
             std::cerr << "Warning: Thread " << i << " got no pages initially." << std::endl;
        }
    }
    const size_t initial_free_pages = alloc.get_num_free_pages();

    std::atomic<bool> start_signal{false};
    std::vector<std::thread> threads;
    std::atomic<size_t> total_ops{0};

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            while (!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); }

            std::vector<uint32_t>& my_pages = thread_local_pages[i];
            if (my_pages.empty()) return; // Skip thread if it got no pages

            for (size_t op = 0; op < ops_per_thread; ++op) {
                const size_t list_idx = op % my_pages.size();
                uint32_t page_to_free = my_pages[list_idx];

                alloc.free_page(page_to_free);

                // Introduce occasional yield to simulate scheduling variance
                if (op % 20 == 0) std::this_thread::yield();

                // Attempt to allocate a new page
                std::optional<uint32_t> new_page_opt;
                int retries = 0;
                constexpr int max_retries = 100; // Limit retries to avoid hangs if pool is stuck empty
                do {
                   new_page_opt = alloc.allocate_page();
                   if(new_page_opt || retries >= max_retries) break;
                   std::this_thread::yield();
                   retries++;
                } while(true);


                if(new_page_opt) {
                    my_pages[list_idx] = *new_page_opt; // Replace freed page in local list
                    total_ops.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Failed to re-allocate, remove the entry from our list
                    // Note: This means the thread's active pages decreases
                    my_pages.erase(my_pages.begin() + list_idx);
                    if(my_pages.empty()) break; // Stop if this thread has no more pages
                }
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& t : threads) {
        t.join();
    }

     std::cout << "HighContentionPushPopStress Stats: "
              << num_threads << " threads, "
              << total_ops.load() << " successful free/alloc pairs completed." << std::endl;

    // Final cleanup: Free all pages held by threads
    size_t pages_freed_in_cleanup = 0;
    for (auto& thread_page_list : thread_local_pages) {
        for (uint32_t page_id : thread_page_list) {
             // Check ref count before freeing, should be 1 if logic is correct
             if(alloc.get_page(page_id).get_ref_count() == 1) {
                 alloc.free_page(page_id);
                 pages_freed_in_cleanup++;
             } else {
                  std::cerr << "Warning: Page " << page_id << " had unexpected ref count before final free." << std::endl;
                  // Attempt free anyway if ref count > 0, might indicate a leak or race condition missed
                  if (alloc.get_page(page_id).get_ref_count() > 0) {
                       // This might over-free if another thread still holds a ref somehow, but unlikely here.
                       // Consider adding a mechanism to track expected refs if debugging complex issues.
                       alloc.free_page(page_id);
                       pages_freed_in_cleanup++;
                  }
             }
        }
    }

    // Verification: All originally allocated pages should now be free
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages)
        << "Final free page count mismatch. Expected " << num_pages
        << ", Got " << alloc.get_num_free_pages()
        << ". Initial free: " << initial_free_pages
        << ", Freed in cleanup: " << pages_freed_in_cleanup;
}
