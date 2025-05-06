#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "sequence/sequence.hpp"
constexpr size_t MAX_TOKENS_PER_DELTA = 16;

constexpr size_t MAX_LOGPROBS_PER_TOKEN = 20;
constexpr size_t MAX_LOGPROBS_PER_DELTA = MAX_TOKENS_PER_DELTA * MAX_LOGPROBS_PER_TOKEN;

namespace pie_core::ipc {

    enum class ResponseSlotState : uint32_t {
        FREE_FOR_CPP_WRITER = 0, // Slot is available for C++ IPCWriter
        CPP_WRITING = 1,         // C++ IPCWriter is writing to this slot
        READY_FOR_PYTHON = 2,    // Slot is ready for Python IPCConsumer to read
        PYTHON_READING = 3       // Python IPCConsumer is reading from this slot
    };

    struct alignas(64) ResponseDeltaSlot {
        std::atomic<ResponseSlotState> state{ResponseSlotState::FREE_FOR_CPP_WRITER};
        uint64_t request_id{0};         // To correlate with the original request
        uint32_t num_tokens_in_delta{0};
        int32_t tokens[MAX_TOKENS_PER_DELTA]{0}; // Batched token IDs
        float logprobs[MAX_TOKENS_PER_DELTA][MAX_LOGPROBS_PER_TOKEN]{0.0f}; // Batched logprobs
        bool is_final_delta{false};
        sequence::FinishReason finish_reason;
    };

    // --- Response Queue ---
    constexpr size_t RESPONSE_QUEUE_NUM_SLOTS = 1024;
    constexpr size_t RESPONSE_QUEUE_SHM_SIZE = RESPONSE_QUEUE_NUM_SLOTS * sizeof(ResponseDeltaSlot);
    constexpr const char* RESPONSE_QUEUE_SHM_NAME = "/pie_response_slots";

    // Control block for the response queue
    struct alignas(64) ResponseQueueControl {
        std::atomic<uint64_t> producer_idx{0}; // Written by C++ IPCWriter
        std::atomic<uint64_t> consumer_idx{0}; // Written by Python's IPCConsumer binding
    };

} // namespace pie_core::ipc
