#include <gtest/gtest.h>
#include "page_allocator.hpp"

using namespace pie_core;

TEST(PageAllocator, ExhaustAndRefill) {
    PageAllocator alloc(4, 4, 16);
    std::vector<uint32_t> ids;
    for (int i = 0; i < 4; ++i) {
        auto id = alloc.allocate_page();
        ASSERT_TRUE(id);
        ids.push_back(*id);
    }
    EXPECT_FALSE(alloc.allocate_page());          // exhausted

    for (auto id : ids) alloc.free_page(id);
    EXPECT_EQ(alloc.get_num_free_pages(), 4);
}
