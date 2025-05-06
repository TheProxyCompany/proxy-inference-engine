#pragma once

#include "sequence/sequence.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"

#include <string>
#include <memory>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

namespace pie_core::ipc {

    // --- IPC Definitions ---
    enum class RequestState : uint32_t {
        FREE = 0,    // Slot is available
        WRITING = 1, // Python is writing to this slot
        READY = 2,   // Slot is ready for C++ to read
        READING = 3  // C++ is reading from this slot
    };

    struct alignas(64) RequestSlot {
        std::atomic<RequestState> state{RequestState::FREE};
        uint64_t request_id{0};
        uint64_t prompt_shm_offset{0};
        uint64_t prompt_shm_size{0};

        sequence::SamplingParams sampling_params;
        sequence::LogitsParams logits_params;
        sequence::StopCriteria stop_criteria;
        sequence::IPCHandles ipc_handles;

        std::string tool_schemas_str;
        std::string response_format_str;
    };

    constexpr size_t REQUEST_QUEUE_NUM_SLOTS = 1024;
    constexpr size_t REQUEST_QUEUE_SHM_SIZE = REQUEST_QUEUE_NUM_SLOTS * sizeof(RequestSlot);
    constexpr const char* REQUEST_QUEUE_SHM_NAME = "/pie_request_slots";

    struct alignas(64) RequestQueueControl {
        std::atomic<uint64_t> producer_idx{0};
        std::atomic<uint64_t> consumer_idx{0};
    };

} // namespace pie_core::ipc
