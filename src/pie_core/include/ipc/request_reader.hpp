#pragma once

#include "sequence/sequence.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"
#include "engine/raw_request.hpp"
#include "ipc/request.hpp"
#include "ipc/shared_memory_manager.hpp"
#include <boost/lockfree/spsc_queue.hpp>

#include <string>
#include <memory>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

namespace pie_core::ipc {

    // --- RequestReader Class ---
    class RequestReader {
    public:

        using RawRequestQueue = boost::lockfree::spsc_queue<
            std::unique_ptr<engine::RawRequestData>,
            boost::lockfree::capacity<1024>
        >;

        RequestReader(
            RawRequestQueue& output_queue,
            SharedMemoryManager& shm_manager,
            const std::string& request_shm_name = REQUEST_QUEUE_SHM_NAME,
            int kernel_event_fd = -1
        );
        ~RequestReader();

        void start();
        void stop();

        RequestReader(const RequestReader&) = delete;
        RequestReader& operator=(const RequestReader&) = delete;
        RequestReader(RequestReader&&) = delete;
        RequestReader& operator=(RequestReader&&) = delete;

    private:
        int request_shm_fd_ = -1;

        void* request_shm_map_ptr_ = nullptr;
        RequestSlot* request_slots_ = nullptr;
        RequestQueueControl* request_queue_control_ = nullptr;

        // --- Bulk Data ---
        int bulk_data_shm_fd_ = -1;
        void* bulk_data_map_ptr_ = nullptr;
        const char* const BULK_DATA_SHM_NAME = "/pie_bulk_data";
        constexpr static size_t BULK_DATA_SHM_SIZE = 1024 * 1024 * 256; // 256MB example

        int kernel_event_fd_;
        std::atomic<bool> running_{false};

        RawRequestQueue& output_queue_;
        SharedMemoryManager& shm_manager_;

        bool initialize_ipc_resources(const std::string& name);
        void cleanup_ipc_resources();
        bool wait_for_notification();
        void process_incoming_requests();
        std::string read_prompt_string(uint64_t offset, uint64_t size);
    };

} // namespace pie_core::ipc
