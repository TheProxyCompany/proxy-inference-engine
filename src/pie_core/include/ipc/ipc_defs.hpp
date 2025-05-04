// src/pie_core/include/ipc_defs.hpp
#pragma once
#include <cstdint>
#include <atomic>

namespace pie_core::ipc {

// Ensure cache-line alignment to prevent false sharing
struct alignas(64) RequestSlot {
    // State: 0=FREE, 1=LOCK (producer claiming), 2=READY (consumer can read)
    std::atomic<uint32_t> state{0};
    uint64_t request_id{0};
    // Add other small metadata fields later (e.g., sampling params)
    // We'll handle large token_ids separately
    // --- Add padding if needed to reach cache line size ---
};

constexpr size_t NUM_SLOTS = 1024; // Power of 2 for efficient ring buffer indexing
constexpr size_t SHM_SIZE = NUM_SLOTS * sizeof(RequestSlot);
const char* const SHM_NAME = "/pie_request_slots"; // Name for shm_open

// We also need an atomic index for the producer(s)
struct alignas(64) RingBufferControl {
     std::atomic<uint64_t> producer_idx{0};
     // Consumer index can often be non-atomic if only one consumer
     // uint64_t consumer_idx{0}; // Managed by consumer thread
};
// Note: This control block could also live at the start of the SHM segment

} // namespace pie_core::ipc
