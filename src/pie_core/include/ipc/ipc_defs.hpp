#pragma once

#include <cstdint>
#include <atomic>
#include <cstddef>
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"

namespace pie_core::ipc {

    struct alignas(64) RequestSlot {
        std::atomic<uint32_t> state{0}; // 0=FREE, 1=LOCKING, 2=READY
        uint64_t request_id{0};
        uint64_t prompt_shm_offset{0}; // Offset in bulk SHM for prompt string
        uint64_t prompt_shm_size{0};   // Size of prompt string in SHM
        SamplingParams sampling_params;
        LogitsParams logits_params;
    };

    constexpr size_t NUM_SLOTS = 1024; // Power of 2 for efficient ring buffer indexing
    constexpr size_t SHM_SIZE = NUM_SLOTS * sizeof(RequestSlot);
    const char* const SHM_NAME = "/pie_request_slots"; // Name for shm_open

    struct alignas(64) RingBufferControl {
        std::atomic<uint64_t> producer_idx{0};
    };

} // namespace pie_core::ipc
