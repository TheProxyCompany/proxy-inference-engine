#include <benchmark/benchmark.h>
#include <engine/page_allocator.hpp>
#include "utils/profiled_page_allocator.hpp"
#include "utils/tracy_wrapper.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace pie_core;

const unsigned int MAX_HARDWARE_THREADS = std::max(1u, std::thread::hardware_concurrency());

#if defined(TRACY_ENABLE)
using BenchAllocatorType = profiling::ProfiledAllocatorWrapper<engine::PageAllocator>;
#else
using BenchAllocatorType = engine::PageAllocator;
#endif

template<typename... Args>
auto create_bench_allocator(Args&&... args) {
    #if defined(TRACY_ENABLE)
    static engine::PageAllocator actual_allocator(std::forward<Args>(args)...);
    return BenchAllocatorType(actual_allocator);
    #else
    return BenchAllocatorType(std::forward<Args>(args)...);
    #endif
}

engine::PageAllocator& get_global_allocator(size_t num_pages, int32_t num_heads, int32_t head_dim) {
    static engine::PageAllocator global_allocator(num_pages, num_heads, head_dim);
    return global_allocator;
}

#if defined(TRACY_ENABLE)
BenchAllocatorType wrap_global_allocator(engine::PageAllocator& allocator) {
    return BenchAllocatorType(allocator);
}
#else
engine::PageAllocator& wrap_global_allocator(engine::PageAllocator& allocator) {
    return allocator;
}
#endif

double calculate_total_memory_mb(size_t num_pages, int32_t num_heads, int32_t head_dim) {
    constexpr double BYTES_PER_MB = 1024.0 * 1024.0;
    const size_t bytes_per_page = engine::TOKEN_CAPACITY_PER_PAGE *
                                  static_cast<size_t>(num_heads) *
                                  static_cast<size_t>(head_dim) *
                                  sizeof(int8_t) * 2;
    return static_cast<double>(num_pages * bytes_per_page) / BYTES_PER_MB;
}

static void BM_PageAllocator_SingleThreadedAllocation(benchmark::State& state) {
    PIE_PROFILE_FUNCTION();

    const size_t num_pages = static_cast<size_t>(state.range(0));
    const int32_t num_heads = static_cast<int32_t>(state.range(1));
    const int32_t head_dim = static_cast<int32_t>(state.range(2));

    state.counters["TotalMemory_MB"] = calculate_total_memory_mb(num_pages, num_heads, head_dim);

    auto bench_allocator = create_bench_allocator(num_pages, num_heads, head_dim);

    std::vector<uint32_t> allocated_pages;
    allocated_pages.reserve(num_pages);

    for (auto _ : state) {
        allocated_pages.clear();

        {
            PIE_PROFILE_ZONE("Allocate All Pages");
            for (size_t i = 0; i < num_pages; ++i) {
                auto page_id = bench_allocator.allocate_page();
                if (!page_id) {
                    state.SkipWithError("Failed to allocate page - pool exhausted");
                    return;
                }
                allocated_pages.push_back(*page_id);
            }
        }
        benchmark::DoNotOptimize(allocated_pages.data());
        benchmark::ClobberMemory();

        {
            PIE_PROFILE_ZONE("Free All Pages");
            for (uint32_t page_id : allocated_pages) {
                try {
                    bench_allocator.free_page(page_id);
                } catch (const std::out_of_range& e) {
                    state.SkipWithError(e.what());
                    return;
                }
            }
        }
        benchmark::DoNotOptimize(allocated_pages.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(num_pages) * 2);
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(num_pages * engine::TOKEN_CAPACITY_PER_PAGE * num_heads * head_dim * sizeof(int8_t) * 2)
    );
}

static void BM_PageAllocator_MultiThreadedAllocation(benchmark::State& state) {
    PIE_PROFILE_FUNCTION();

    const size_t num_pages = static_cast<size_t>(state.range(0));
    const int32_t num_heads = static_cast<int32_t>(state.range(2));
    const int32_t head_dim = static_cast<int32_t>(state.range(3));
    const int actual_threads = state.threads();

    engine::PageAllocator& global_allocator = get_global_allocator(num_pages, num_heads, head_dim);

    state.counters["TotalMemory_MB"] = calculate_total_memory_mb(num_pages, num_heads, head_dim);
    state.counters["ThreadCount"] = actual_threads;

    const size_t pages_per_thread = num_pages / actual_threads;
    const size_t extra_pages = num_pages % actual_threads;
    const size_t start_idx = static_cast<size_t>(state.thread_index()) * pages_per_thread + std::min(static_cast<size_t>(state.thread_index()), extra_pages);
    const size_t end_idx = start_idx + pages_per_thread + (static_cast<size_t>(state.thread_index()) < extra_pages ? 1 : 0);
    const size_t thread_page_count = end_idx - start_idx;

    for (auto _ : state) {
        PIE_PROFILE_ZONE("Multithreaded Allocation Iteration");
        PIE_PROFILE_THREAD(("AllocatorWorker-" + std::to_string(state.thread_index())).c_str());

        #if defined(TRACY_ENABLE)
        BenchAllocatorType bench_allocator = wrap_global_allocator(global_allocator);
        #else
        engine::PageAllocator& bench_allocator = wrap_global_allocator(global_allocator);
        #endif

        std::vector<uint32_t> thread_pages;
        thread_pages.reserve(thread_page_count);

        {
            PIE_PROFILE_ZONE("Thread Allocation");
             for (size_t i = 0; i < thread_page_count; ++i) {
                auto page_id = bench_allocator.allocate_page();
                if (page_id) {
                    thread_pages.push_back(*page_id);
                 }
            }
        }
        benchmark::DoNotOptimize(thread_pages.data());
        benchmark::ClobberMemory();

        {
            PIE_PROFILE_ZONE("Thread Deallocation");
            for (uint32_t page_id : thread_pages) {
                 try {
                    bench_allocator.free_page(page_id);
                } catch (const std::out_of_range& e) {
                    state.SkipWithError(e.what());
                    return;
                }
            }
        }
        benchmark::DoNotOptimize(thread_pages.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(num_pages) * 2);
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(num_pages * engine::TOKEN_CAPACITY_PER_PAGE * num_heads * head_dim * sizeof(int8_t) * 2)
    );
}

static void BM_PageAllocator_ReferenceCountingScenario(benchmark::State& state) {
    PIE_PROFILE_FUNCTION();

    const size_t num_pages = static_cast<size_t>(state.range(0));
    const int32_t num_heads = static_cast<int32_t>(state.range(1));
    const int32_t head_dim = static_cast<int32_t>(state.range(2));

    state.counters["TotalMemory_MB"] = calculate_total_memory_mb(num_pages, num_heads, head_dim);
    state.counters["RefCountingScenario"] = 1;

    auto bench_allocator = create_bench_allocator(num_pages, num_heads, head_dim);

    std::vector<uint32_t> allocated_pages;
    allocated_pages.reserve(num_pages);
    std::vector<uint32_t> pages_to_re_release;
    pages_to_re_release.reserve(num_pages / 4);

    const size_t phase1_alloc_count = num_pages / 2;
    const size_t phase2_ref_count = phase1_alloc_count / 2;

    for (auto _ : state) {
        allocated_pages.clear();
        pages_to_re_release.clear();

        {
            PIE_PROFILE_ZONE("Phase1: Initial Allocation");
            for (size_t i = 0; i < phase1_alloc_count; ++i) {
                auto page_id = bench_allocator.allocate_page();
                if (!page_id) {
                    state.SkipWithError("Failed Phase 1 allocation");
                    return;
                }
                allocated_pages.push_back(*page_id);
            }
        }
        benchmark::DoNotOptimize(allocated_pages.data());
        benchmark::ClobberMemory();

        {
            PIE_PROFILE_ZONE("Phase2: Add References");
            for (size_t i = 0; i < phase2_ref_count; ++i) {
                 try {
                    bench_allocator.add_ref(allocated_pages[i]);
                    pages_to_re_release.push_back(allocated_pages[i]);
                 } catch (const std::out_of_range& e) {
                     state.SkipWithError(e.what());
                     return;
                 }
            }
        }
        benchmark::ClobberMemory();

        {
            PIE_PROFILE_ZONE("Phase3: First Free Attempt");
            for (uint32_t page_id : allocated_pages) {
                 try {
                     bench_allocator.free_page(page_id);
                 } catch (const std::out_of_range& e) {
                     state.SkipWithError(e.what());
                     return;
                 }
            }
        }
        benchmark::ClobberMemory();

        size_t reallocated_count = 0;
        {
            PIE_PROFILE_ZONE("Phase4: Reallocate Available Pages");
            size_t max_possible_reallocs = num_pages - phase2_ref_count;
            for (size_t i = 0; i < max_possible_reallocs; ++i) {
                auto page_id = bench_allocator.allocate_page();
                if (!page_id) {
                    break;
                }
                 reallocated_count++;
            }
            state.counters["PagesReallocated"] = static_cast<double>(reallocated_count);
        }

        {
             PIE_PROFILE_ZONE("Phase5: Release Extra References");
             for (uint32_t page_id : pages_to_re_release) {
                 try {
                     bench_allocator.free_page(page_id);
                 } catch (const std::out_of_range& e) {
                     state.SkipWithError(e.what());
                     return;
                 }
             }
        }
        benchmark::ClobberMemory();

        {
             PIE_PROFILE_ZONE("Phase6: Final Cleanup");
             for (uint32_t page_id : allocated_pages) {
                 try {
                     bench_allocator.free_page(page_id);
                 } catch (const std::exception&) {
                     // Ignore expected errors during cleanup
                 }
             }
        }
         benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(num_pages) * 3);
     state.SetBytesProcessed(
         static_cast<int64_t>(state.iterations()) *
         static_cast<int64_t>(num_pages * engine::TOKEN_CAPACITY_PER_PAGE * num_heads * head_dim * sizeof(int8_t) * 2)
     );
}

static void AddModelSizeArgs(benchmark::internal::Benchmark* b) {
    b->Args({2000, 32, 80});   // ~3B params
    b->Args({5000, 32, 128});  // ~7B params
    b->Args({10000, 40, 128}); // ~13B params
    b->Args({20000, 60, 128}); // ~30B+ params
}

static void BM_PageAllocator_SimulateLLMInference(benchmark::State& state) {
    PIE_PROFILE_FUNCTION();

    const size_t num_pages = static_cast<size_t>(state.range(0));
    const int32_t num_heads = static_cast<int32_t>(state.range(1));
    const int32_t head_dim = static_cast<int32_t>(state.range(2));
    const int sequence_length = static_cast<int>(state.range(4));
    const int benchmark_threads = state.threads();

    engine::PageAllocator& global_allocator = get_global_allocator(num_pages, num_heads, head_dim);

    struct Session {
        std::vector<uint32_t> pages;
        int tokens_generated = 0;
        bool completed = false;
    };

    struct IterationState {
        std::vector<Session> sessions;
        std::mutex mutex;
        std::atomic<int> active_sessions_count;
        std::atomic<bool> stop_flag{false};

         IterationState(int num_simulated_sessions)
             : sessions(num_simulated_sessions), active_sessions_count(num_simulated_sessions) {}
    };

    IterationState shared_iteration_state(benchmark_threads);

    state.counters["TotalMemory_MB"] = calculate_total_memory_mb(num_pages, num_heads, head_dim);
    state.counters["BenchmarkThreads"] = benchmark_threads;
    state.counters["SimulatedSeqLen"] = sequence_length;
    state.counters["SimulatedSessions"] = benchmark_threads;

    constexpr auto SIMULATION_TIMEOUT = std::chrono::seconds(10);

    for (auto _ : state) {
        shared_iteration_state.stop_flag.store(false, std::memory_order_relaxed);
        shared_iteration_state.active_sessions_count.store(benchmark_threads, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(shared_iteration_state.mutex);
            for (auto& session : shared_iteration_state.sessions) {
                session.pages.clear();
                session.tokens_generated = 0;
                session.completed = false;
            }
        }

        PIE_PROFILE_ZONE("Inference Simulation Iteration");
        PIE_PROFILE_THREAD(("LLM-SimWorker-" + std::to_string(state.thread_index())).c_str());

        #if defined(TRACY_ENABLE)
        BenchAllocatorType thread_allocator = wrap_global_allocator(global_allocator);
        #else
        engine::PageAllocator& thread_allocator = wrap_global_allocator(global_allocator);
        #endif
        int thread_idx = state.thread_index();

        auto start_time = std::chrono::steady_clock::now();

        while (!shared_iteration_state.stop_flag.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() - start_time > SIMULATION_TIMEOUT) {
                state.SkipWithError("Benchmark timeout during simulation");
                shared_iteration_state.stop_flag.store(true, std::memory_order_release);
                break;
            }

            bool just_completed = false;
            {
                std::lock_guard<std::mutex> lock(shared_iteration_state.mutex);

                 assert(thread_idx >= 0 && static_cast<size_t>(thread_idx) < shared_iteration_state.sessions.size());
                 Session& session = shared_iteration_state.sessions[thread_idx];

                if (session.completed) {
                    break;
                }

                session.tokens_generated++;

                if (session.pages.empty() || (session.tokens_generated > 0 && session.tokens_generated % engine::TOKEN_CAPACITY_PER_PAGE == 1)) {
                     PIE_PROFILE_ZONE("Allocate New Page");
                     auto page_id = thread_allocator.allocate_page();
                    if (page_id) {
                        session.pages.push_back(*page_id);
                        if (session.pages.size() > 2) {
                             try {
                                 thread_allocator.add_ref(session.pages[0]);
                             } catch (const std::out_of_range& e) {
                                 state.SkipWithError(e.what());
                                 shared_iteration_state.stop_flag.store(true, std::memory_order_release);
                                 break;
                             }
                         }
                    } else {
                         state.SkipWithError("Failed to allocate page during simulation");
                         shared_iteration_state.stop_flag.store(true, std::memory_order_release);
                         break;
                    }
                 }

                 if (session.tokens_generated >= sequence_length) {
                    session.completed = true;
                    just_completed = true;
                 }
            }

             if (just_completed) {
                 int remaining = shared_iteration_state.active_sessions_count.fetch_sub(1, std::memory_order_acq_rel);
                 if (remaining - 1 == 0) {
                     shared_iteration_state.stop_flag.store(true, std::memory_order_release);
                 }
                 break;
             }

             if (!shared_iteration_state.stop_flag.load(std::memory_order_acquire)) {
                 std::this_thread::sleep_for(std::chrono::microseconds(1));
             }
        }

        {
            PIE_PROFILE_ZONE("Session Cleanup");
             std::lock_guard<std::mutex> lock(shared_iteration_state.mutex);

             assert(thread_idx >= 0 && static_cast<size_t>(thread_idx) < shared_iteration_state.sessions.size());
             Session& session = shared_iteration_state.sessions[thread_idx];

             if (!session.pages.empty()) {
                 int num_pages_allocated = static_cast<int>(session.pages.size());
                 int extra_refs_added = std::max(0, num_pages_allocated - 2);

                 for (int i = 0; i < extra_refs_added; ++i) {
                     try {
                         thread_allocator.free_page(session.pages[0]);
                     } catch (const std::exception& e) {
                         break;
                     }
                 }

                 for (uint32_t page_id : session.pages) {
                    try {
                        thread_allocator.free_page(page_id);
                    } catch (const std::exception& e) {
                         // Tolerate errors during general cleanup phase
                    }
                }
             }

             session.pages.clear();
             session.tokens_generated = 0;
             session.completed = true;
        }

         if (state.thread_index() == 0) {
             #if defined(TRACY_ENABLE)
             state.counters["MemoryUtilization_Percent"] = wrap_global_allocator(global_allocator).get_memory_utilization_percent();
             #else
             size_t total_pages = global_allocator.size();
             size_t free_pages = global_allocator.get_num_free_pages();
             size_t used_pages = (total_pages >= free_pages) ? total_pages - free_pages : 0;
             double percentage = (total_pages > 0)
                 ? (static_cast<double>(used_pages) / total_pages) * 100.0
                 : 0.0;
             state.counters["MemoryUtilization_Percent"] = percentage;
             #endif
         }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(benchmark_threads) * static_cast<int64_t>(sequence_length));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(num_pages * engine::TOKEN_CAPACITY_PER_PAGE * num_heads * head_dim * sizeof(int8_t) * 2)
    );
}

BENCHMARK(BM_PageAllocator_SingleThreadedAllocation)
    ->Apply(AddModelSizeArgs)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PageAllocator_MultiThreadedAllocation)
    ->Args({2000, 0, 32, 80}) ->Threads(1)->Threads(2)->Threads(4)->Threads(std::min(8u, MAX_HARDWARE_THREADS))
    ->Args({5000, 0, 32, 128}) ->Threads(1)->Threads(2)->Threads(4)->Threads(8)->Threads(std::min(16u, MAX_HARDWARE_THREADS))
    ->Args({10000, 0, 40, 128})->Threads(1)->Threads(2)->Threads(4)->Threads(8)->Threads(std::min(16u, MAX_HARDWARE_THREADS))
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_PageAllocator_ReferenceCountingScenario)
    ->Apply(AddModelSizeArgs)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_PageAllocator_SimulateLLMInference)
    ->Args({4000, 32, 80, 0, 512})  ->Threads(std::min(4u, MAX_HARDWARE_THREADS))
    ->Args({4000, 32, 80, 0, 1024}) ->Threads(std::min(4u, MAX_HARDWARE_THREADS))
    ->Args({4000, 32, 80, 0, 1024}) ->Threads(std::min(8u, MAX_HARDWARE_THREADS))
    ->Args({8000, 32, 128, 0, 1024})->Threads(std::min(4u, MAX_HARDWARE_THREADS))
    ->Args({8000, 32, 128, 0, 1024})->Threads(std::min(8u, MAX_HARDWARE_THREADS))
    ->Args({16000, 40, 128, 0, 1024})->Threads(std::min(4u, MAX_HARDWARE_THREADS))
    ->Args({16000, 40, 128, 0, 1024})->Threads(std::min(8u, MAX_HARDWARE_THREADS))
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // anonymous namespace
