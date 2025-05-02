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

using namespace pie_core;

// --- Test Fixture for Convenience ---
class PageAllocatorTest : public ::testing::Test {
protected:
    const int32_t DEFAULT_NUM_HEADS = 4;
    const int32_t DEFAULT_HEAD_DIM = 16;
};


// --- Basic Allocation and Freeing Tests ---

TEST_F(PageAllocatorTest, BasicExhaustAndRefill) {
    const size_t num_pages = 4;
    PageAllocator alloc(num_pages, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.size(), num_pages);
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages);

    std::vector<uint32_t> ids;
    for (size_t i = 0; i < num_pages; ++i) {
        auto id_opt = alloc.allocate_page();
        ASSERT_TRUE(id_opt.has_value()) << "Allocation failed at iteration " << i;
        ids.push_back(*id_opt);
        EXPECT_EQ(alloc.get_num_free_pages(), num_pages - (i + 1));
        EXPECT_EQ(alloc.get_page(*id_opt).get_ref_count(), 1); // Check initial ref count
    }

    // Check exhaustion
    EXPECT_FALSE(alloc.allocate_page().has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), 0);

    // Free pages
    for (size_t i = 0; i < num_pages; ++i) {
        alloc.free_page(ids[i]);
        EXPECT_EQ(alloc.get_num_free_pages(), i + 1);
    }

    // Check if fully refilled
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages);

    // Can allocate again
    auto id_opt_after = alloc.allocate_page();
    ASSERT_TRUE(id_opt_after.has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages - 1);
}

TEST_F(PageAllocatorTest, EdgeCaseSizeOne) {
    PageAllocator alloc(1, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.size(), 1);
    EXPECT_EQ(alloc.get_num_free_pages(), 1);

    auto id1_opt = alloc.allocate_page();
    ASSERT_TRUE(id1_opt.has_value());
    EXPECT_EQ(*id1_opt, 0); // Should be the first page
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_FALSE(alloc.allocate_page().has_value()); // Exhausted

    alloc.free_page(*id1_opt);
    EXPECT_EQ(alloc.get_num_free_pages(), 1);

    auto id2_opt = alloc.allocate_page();
    ASSERT_TRUE(id2_opt.has_value());
    EXPECT_EQ(*id2_opt, 0); // Should get the same page back
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}

// --- LIFO (Stack Behavior) Test ---

TEST_F(PageAllocatorTest, LIFOCheck) {
    const size_t num_pages = 10;
    PageAllocator alloc(num_pages, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    std::vector<uint32_t> allocated_ids;
    for (size_t i = 0; i < num_pages; ++i) {
        allocated_ids.push_back(*alloc.allocate_page());
    }
    EXPECT_EQ(alloc.get_num_free_pages(), 0);

    // Free in reverse order of allocation
    std::vector<uint32_t> freed_ids_order = allocated_ids;
    std::reverse(freed_ids_order.begin(), freed_ids_order.end());
    for (uint32_t id : freed_ids_order) {
        alloc.free_page(id);
    }
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages);

    // Allocate again and check if they come out in the same order they were freed (LIFO)
    std::vector<uint32_t> reallocated_ids;
    for (size_t i = 0; i < num_pages; ++i) {
        reallocated_ids.push_back(*alloc.allocate_page());
    }
    EXPECT_EQ(reallocated_ids, allocated_ids);
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}


// --- Reference Counting Tests ---

TEST_F(PageAllocatorTest, SingleThreadRefCounting) {
    PageAllocator alloc(1, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);

    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t page_id = *id_opt;

    // Check initial state
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    // Add references
    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 2);
    alloc.add_ref(page_id);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 3);

    // Free once - should not return to pool
    alloc.free_page(page_id);
    EXPECT_EQ(alloc.get_num_free_pages(), 0); // Still 0 free
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 2);

    // Free again - should not return to pool
    alloc.free_page(page_id);
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_EQ(alloc.get_page(page_id).get_ref_count(), 1);

    // Final free - should return to pool
    alloc.free_page(page_id);
    EXPECT_EQ(alloc.get_num_free_pages(), 1);
}

// --- Error Handling Tests ---

TEST_F(PageAllocatorTest, InvalidIdThrows) {
    PageAllocator alloc(5, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);

    EXPECT_THROW(alloc.get_page(5), std::out_of_range);
    EXPECT_THROW(alloc.get_page(100), std::out_of_range);
    EXPECT_THROW(alloc.free_page(5), std::out_of_range);
    EXPECT_THROW(alloc.add_ref(5), std::out_of_range);

    // Allocate one page to test freeing/adding ref to valid but potentially busy page
    auto id_opt = alloc.allocate_page();
    ASSERT_TRUE(id_opt.has_value());
    uint32_t valid_id = *id_opt;
    EXPECT_NO_THROW(alloc.get_page(valid_id));
    EXPECT_NO_THROW(alloc.add_ref(valid_id));
    EXPECT_NO_THROW(alloc.free_page(valid_id)); // Ref count is 2 now
    EXPECT_NO_THROW(alloc.free_page(valid_id)); // Ref count is 1 now
}

// --- Multi-threading Tests ---

TEST_F(PageAllocatorTest, ConcurrentAllocFree) {
    const size_t num_pages = 1024; // Use a larger pool for concurrency tests
    const size_t num_threads = std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() : 2; // Use available cores
    const size_t pages_per_producer = num_pages / (num_threads - 1); // -1 for the consumer

    PageAllocator alloc(num_pages, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM);
    EXPECT_EQ(alloc.get_num_free_pages(), num_pages);

    // Pre-allocate pages to be freed by producers
    std::vector<std::vector<uint32_t>> producer_pages(num_threads - 1);
    for(size_t i = 0; i < num_threads - 1; ++i) {
        for(size_t j = 0; j < pages_per_producer; ++j) {
            auto id_opt = alloc.allocate_page();
            ASSERT_TRUE(id_opt.has_value()) << "Pre-allocation failed";
            producer_pages[i].push_back(*id_opt);
        }
    }
     // Allocate any remaining pages if num_pages wasn't perfectly divisible
    while(auto id_opt = alloc.allocate_page()) {
        producer_pages[0].push_back(*id_opt); // Give extras to first producer
    }
    EXPECT_EQ(alloc.get_num_free_pages(), 0); // Pool should be empty now


    std::vector<std::thread> threads;
    std::atomic<size_t> consumer_alloc_count{0};
    std::atomic<bool> start_signal{false};
    std::vector<std::optional<uint32_t>> consumer_allocations(num_pages); // Store results

    // Consumer thread
    threads.emplace_back([&]() {
        while(!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); } // Spin wait for start
        for(size_t i = 0; i < num_pages; ++i) {
            std::optional<uint32_t> id_opt;
            // Spin-wait briefly if allocator is temporarily empty, simulating scheduler waiting
            int retries = 0;
            do {
                id_opt = alloc.allocate_page();
                if (!id_opt && retries < 1000) { // Avoid infinite loop if producers are slow/stuck
                     std::this_thread::yield();
                     retries++;
                } else {
                     break;
                }
            } while(true);

            if(id_opt) {
                consumer_allocations[consumer_alloc_count.fetch_add(1, std::memory_order_relaxed)] = id_opt;
            } else {
                 // Failed to allocate after retries, log or break
                 break;
            }
        }
    });

    // Producer threads
    for (size_t i = 0; i < num_threads - 1; ++i) {
        threads.emplace_back([&, i]() {
            while(!start_signal.load(std::memory_order_acquire)) { std::this_thread::yield(); } // Spin wait for start
            for (uint32_t page_id : producer_pages[i]) {
                alloc.free_page(page_id);
            }
        });
    }

    // Start all threads roughly together
    start_signal.store(true, std::memory_order_release);

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // --- Assertions ---
    EXPECT_EQ(consumer_alloc_count.load(), num_pages) << "Consumer did not allocate all pages";
    EXPECT_EQ(alloc.get_num_free_pages(), 0) << "Allocator should be empty after consumer finishes";

    // Check if all allocated IDs are unique (optional but good sanity check)
    std::set<uint32_t> unique_ids;
    size_t valid_allocs = 0;
    for(const auto& id_opt : consumer_allocations) {
        if (id_opt) {
            unique_ids.insert(*id_opt);
            valid_allocs++;
        }
    }
     EXPECT_EQ(valid_allocs, num_pages) << "Number of valid allocations mismatch";
    EXPECT_EQ(unique_ids.size(), num_pages) << "Allocated page IDs were not unique";
}
