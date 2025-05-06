#pragma once

#include "sequence/sequence.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"
#include "ipc/ipc_request.hpp"

#include <string>
#include <memory>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

namespace pie_core::ipc {
    template<typename T>
    class SPSCQueue;
}

namespace pie_core::ipc {

    // --- IPCReader Class ---
    class IPCReader {
    public:
        using SequenceQueueType = SPSCQueue<std::unique_ptr<sequence::Sequence>>;

        IPCReader(
            SequenceQueueType& output_queue,
            const std::string& request_shm_name = REQUEST_QUEUE_SHM_NAME,
            int kernel_event_fd = -1
        );
        ~IPCReader();

        void run();
        void stop();

        IPCReader(const IPCReader&) = delete;
        IPCReader& operator=(const IPCReader&) = delete;
        IPCReader(IPCReader&&) = delete;
        IPCReader& operator=(IPCReader&&) = delete;

    private:
        int request_shm_fd_ = -1;
        void* request_shm_map_ptr_ = nullptr;
        RequestSlot* request_slots_ = nullptr;
        RequestQueueControl* request_queue_control_ = nullptr;

        // For bulk data (e.g., prompts)
        // Simplification: Assume one primary bulk SHM segment.
        // If multiple are needed, this needs more complex management.
        int bulk_data_shm_fd_ = -1;
        void* bulk_data_map_ptr_ = nullptr;
        const char* const BULK_DATA_SHM_NAME = "/pie_bulk_data"; // Example name
        constexpr static size_t BULK_DATA_SHM_SIZE = 1024 * 1024 * 256; // 256MB example

        int kernel_event_fd_;
        std::atomic<bool> running_{false};

        SequenceQueueType& output_queue_;

        bool initialize_ipc_resources();
        void cleanup_ipc_resources();
        bool wait_for_notification();
        void process_incoming_requests();
        std::unique_ptr<sequence::Sequence> build_sequence_from_slot(const RequestSlot& slot);
        std::vector<int32_t> read_prompt_tokens(uint64_t offset, uint64_t size);
    };

} // namespace pie_core::ipc
