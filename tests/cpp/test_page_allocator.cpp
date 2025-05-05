#include <gtest/gtest.h>
#include "engine/page_allocator.hpp"
#include <vector>
#include <numeric>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>
#include <algorithm>
#include <iterator>

using namespace pie_core;

// -----------------------------------------------------------------------------
// Test fixture holding common constants
// -----------------------------------------------------------------------------
class PageAllocatorTest : public ::testing::Test {
public:
    static constexpr int32_t DEFAULT_NUM_HEADS = 4;
    static constexpr int32_t DEFAULT_HEAD_DIM  = 16;
    static constexpr size_t  LARGE_POOL_SIZE   = 1024;
    static constexpr size_t  SMALL_POOL_SIZE   = 10;
    static constexpr size_t  TINY_POOL_SIZE    = 4;
    static constexpr size_t  SINGLE_PAGE_POOL  = 1;
};

// --------------------------------------------------------------------------
// Small helpers to keep individual tests concise
// --------------------------------------------------------------------------
namespace {

engine::PageAllocator make_allocator(size_t pages) {
    return engine::PageAllocator(pages,
                         PageAllocatorTest::DEFAULT_NUM_HEADS,
                         PageAllocatorTest::DEFAULT_HEAD_DIM);
}

[[nodiscard]] std::vector<uint32_t> allocate_pages(engine::PageAllocator &alloc,
                                                   size_t n) {
    std::vector<uint32_t> ids;
    ids.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto id = alloc.allocate_page();
        EXPECT_TRUE(id.has_value()) << "alloc failed @" << i;
        ids.push_back(*id);
    }
    return ids;
}

void free_pages(engine::PageAllocator &alloc, const std::vector<uint32_t> &ids) {
    for (auto id : ids) alloc.free_page(id);
}

} // namespace

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------
TEST_F(PageAllocatorTest, ConstructorValidArgs) {
    EXPECT_NO_THROW(make_allocator(TINY_POOL_SIZE));
}

TEST_F(PageAllocatorTest, ConstructorInvalidArgs) {
    EXPECT_THROW(engine::PageAllocator(0, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM),
                 std::invalid_argument);
    EXPECT_THROW(engine::PageAllocator(TINY_POOL_SIZE, 0, DEFAULT_HEAD_DIM),
                 std::invalid_argument);
    EXPECT_THROW(engine::PageAllocator(TINY_POOL_SIZE, DEFAULT_NUM_HEADS, 0),
                 std::invalid_argument);
}

// --------------------------------------------------------------------------
// Basic allocation behaviour
// --------------------------------------------------------------------------
TEST_F(PageAllocatorTest, ExhaustAndRefill) {
    auto alloc = make_allocator(TINY_POOL_SIZE);
    EXPECT_EQ(alloc.size(), TINY_POOL_SIZE);
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE);

    const auto ids = allocate_pages(alloc, TINY_POOL_SIZE);
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
    EXPECT_FALSE(alloc.allocate_page().has_value());

    free_pages(alloc, ids);
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE);

    EXPECT_TRUE(alloc.allocate_page().has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), TINY_POOL_SIZE - 1);
}

TEST_F(PageAllocatorTest, ExhaustionReturnsNullopt) {
    auto alloc = make_allocator(2);
    ASSERT_TRUE(alloc.allocate_page().has_value());
    ASSERT_TRUE(alloc.allocate_page().has_value());
    EXPECT_FALSE(alloc.allocate_page().has_value());
    EXPECT_EQ(alloc.get_num_free_pages(), 0);
}

TEST_F(PageAllocatorTest, EdgeCaseSinglePage) {
    auto alloc = make_allocator(SINGLE_PAGE_POOL);
    EXPECT_EQ(alloc.size(), SINGLE_PAGE_POOL);

    auto id = alloc.allocate_page();
    ASSERT_TRUE(id);
    EXPECT_EQ(*id, 0u);
    EXPECT_FALSE(alloc.allocate_page());

    alloc.free_page(*id);
    EXPECT_EQ(alloc.get_num_free_pages(), 1u);

    auto id2 = alloc.allocate_page();
    ASSERT_TRUE(id2);
    EXPECT_EQ(*id2, 0u);
}

// -----------------------------------------------------------------------------
// LIFO behaviour (Treiber stack)
// -----------------------------------------------------------------------------
TEST_F(PageAllocatorTest, LifoOrder) {
    auto alloc        = make_allocator(SMALL_POOL_SIZE);
    const auto first  = allocate_pages(alloc, SMALL_POOL_SIZE);
    std::vector<uint32_t> reversed(first.rbegin(), first.rend());

    free_pages(alloc, reversed);
    EXPECT_EQ(alloc.get_num_free_pages(), SMALL_POOL_SIZE);

    const auto second = allocate_pages(alloc, SMALL_POOL_SIZE);
    EXPECT_EQ(second, first);
    EXPECT_EQ(alloc.get_num_free_pages(), 0u);
}

// -----------------------------------------------------------------------------
// Reference counting
// -----------------------------------------------------------------------------
TEST_F(PageAllocatorTest, SingleThreadRefCounting) {
    auto alloc = make_allocator(SINGLE_PAGE_POOL);
    const auto id  = *alloc.allocate_page();

    EXPECT_EQ(alloc.get_page(id).get_ref_count(), 1u);
    alloc.add_ref(id);
    alloc.add_ref(id);
    EXPECT_EQ(alloc.get_page(id).get_ref_count(), 3u);

    alloc.free_page(id);
    alloc.free_page(id);
    EXPECT_EQ(alloc.get_page(id).get_ref_count(), 1u);
    EXPECT_EQ(alloc.get_num_free_pages(), 0u);

    alloc.free_page(id);
    EXPECT_EQ(alloc.get_num_free_pages(), 1u);
}

TEST_F(PageAllocatorTest, ExplicitAddRef) {
    auto alloc = make_allocator(SINGLE_PAGE_POOL);
    const auto id = *alloc.allocate_page();

    alloc.add_ref(id);
    EXPECT_EQ(alloc.get_page(id).get_ref_count(), 2u);

    EXPECT_THROW(alloc.add_ref(999), std::out_of_range);
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------
TEST_F(PageAllocatorTest, GetPageNonConst) {
    auto alloc = make_allocator(2);
    const auto id1 = *alloc.allocate_page();
    const auto id2 = *alloc.allocate_page();

    EXPECT_EQ(alloc.get_page(id1).page_id(), id1);
    EXPECT_EQ(alloc.get_page(id2).page_id(), id2);
    EXPECT_NO_THROW(alloc.get_page(id1).key_cache());
    EXPECT_THROW(alloc.get_page(999), std::out_of_range);
}

TEST_F(PageAllocatorTest, GetPageConst) {
    auto alloc = make_allocator(2);
    const auto id1 = *alloc.allocate_page();
    const auto id2 = *alloc.allocate_page();
    const auto &c  = alloc;

    EXPECT_EQ(c.get_page(id1).page_id(), id1);
    EXPECT_EQ(c.get_page(id2).page_id(), id2);
    EXPECT_EQ(c.get_page(id1).get_ref_count(), 1u);
    EXPECT_THROW(c.get_page(999), std::out_of_range);
}

// -----------------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------------
TEST_F(PageAllocatorTest, InvalidIdThrows) {
    auto alloc = make_allocator(5);
    EXPECT_THROW(alloc.get_page(5),  std::out_of_range);
    EXPECT_THROW(alloc.get_page(100), std::out_of_range);
    EXPECT_THROW(alloc.free_page(5),  std::out_of_range);
    EXPECT_THROW(alloc.add_ref(5),    std::out_of_range);
}

// -----------------------------------------------------------------------------
// Concurrency – keep tests verbose for clarity, but remove noise
// -----------------------------------------------------------------------------
TEST_F(PageAllocatorTest, ConcurrentAllocFreeProducersConsumer) {
    const size_t num_threads   = std::max(2u, std::thread::hardware_concurrency());
    const size_t num_producers = num_threads - 1;
    const size_t pages_per_p   = LARGE_POOL_SIZE / num_producers;
    const size_t total_pages   = pages_per_p * num_producers;

    auto alloc = make_allocator(total_pages);

    // Pre-allocate pages for producers to release.
    std::vector<std::vector<uint32_t>> initial(num_producers);
    for (auto &v : initial) v = allocate_pages(alloc, pages_per_p);
    ASSERT_EQ(alloc.get_num_free_pages(), 0u);

    std::atomic<size_t> alloc_count{0};
    std::atomic<bool>   start{false};
    std::vector<std::optional<uint32_t>> consumer_ids(total_pages);

    std::vector<std::thread> threads;
    // Consumer
    threads.emplace_back([&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        for (size_t i = 0; i < total_pages; ++i) {
            std::optional<uint32_t> id;
            for (size_t retry = 0; retry < 5000 && !(id = alloc.allocate_page()); ++retry)
                std::this_thread::yield();
            ASSERT_TRUE(id) << "consumer failed @" << i;
            consumer_ids[alloc_count.fetch_add(1)] = id;
        }
    });

    // Producers
    for (size_t p = 0; p < num_producers; ++p) {
        threads.emplace_back([&, p] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (auto page_id : initial[p]) alloc.free_page(page_id);
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();

    EXPECT_EQ(alloc_count.load(), total_pages);
    EXPECT_EQ(alloc.get_num_free_pages(), 0u);

    std::set<uint32_t> uniq;
    for (const auto &opt : consumer_ids) EXPECT_TRUE(uniq.insert(*opt).second);
    EXPECT_EQ(uniq.size(), total_pages);
}

TEST_F(PageAllocatorTest, ConcurrentFreeSharedPage) {
    constexpr int refs = 10;
    auto alloc = make_allocator(SINGLE_PAGE_POOL);
    const auto id = *alloc.allocate_page();

    for (int i = 1; i < refs; ++i) alloc.add_ref(id);
    EXPECT_EQ(alloc.get_page(id).get_ref_count(), refs);

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < refs; ++i) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            alloc.free_page(id);
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();

    EXPECT_EQ(alloc.get_num_free_pages(), 1u);
    EXPECT_EQ(*alloc.allocate_page(), id);
    EXPECT_EQ(alloc.get_page(id).get_ref_count(), 1u);
}

// Intentionally more verbose – stresses push/pop under heavy contention.
TEST_F(PageAllocatorTest, HighContentionPushPopStress) {
    constexpr size_t num_pages   = 128;
    const size_t num_threads     = std::max(4u, std::thread::hardware_concurrency());
    constexpr size_t ops_per_thr = 2000;

    auto alloc = make_allocator(num_pages);
    const size_t pages_per_thread = 4;

    std::vector<std::vector<uint32_t>> local(num_threads);
    for (auto &v : local) v = allocate_pages(alloc, pages_per_thread);
    const size_t initial_free = alloc.get_num_free_pages();

    std::atomic<bool> start{false};
    std::atomic<size_t> total_ops{0};
    std::vector<std::thread> threads;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([&, tid] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            auto &owned = local[tid];
            for (size_t op = 0; op < ops_per_thr && !owned.empty(); ++op) {
                const size_t idx = op % owned.size();
                alloc.free_page(owned[idx]);
                std::optional<uint32_t> np;
                for (int r = 0; r < 100 && !(np = alloc.allocate_page()); ++r)
                    std::this_thread::yield();
                if (np) {
                    owned[idx] = *np;
                    total_ops.fetch_add(1);
                } else {
                    owned.erase(owned.begin() + idx);
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();

    for (const auto &vec : local)
        for (auto id : vec)
            if (alloc.get_page(id).get_ref_count() == 1) alloc.free_page(id);

    EXPECT_EQ(alloc.get_num_free_pages(), num_pages)
        << "leak detected, initial_free=" << initial_free;
}
