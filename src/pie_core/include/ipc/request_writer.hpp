#pragma once

#include "request.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "ipc/shared_memory_manager.hpp"
#include <string>
#include <vector>
#include <cstdint>

// exposed via nanobind to python
namespace pie_core::ipc {

    constexpr const char* BULK_DATA_SHM_NAME = "/pie_bulk_data";
    constexpr size_t BULK_DATA_SHM_SIZE = 1024 * 1024 * 256; // 256MB

    class RequestWriter {
    public:
        RequestWriter(
            const std::string& request_shm_name = REQUEST_QUEUE_SHM_NAME,
            const std::string& bulk_shm_name = BULK_DATA_SHM_NAME
        );
        ~RequestWriter();

        uint64_t submit_request_to_engine(
            uint64_t request_id,
            const std::string& prompt_string,
            const sequence::SamplingParams& sampling_params,
            const sequence::LogitsParams& logits_params,
            const sequence::StopCriteria& stop_criteria,
            const sequence::IPCHandles& ipc_handles,
            const std::string& tool_schemas_str,
            const std::string& response_format_str
        );

        RequestWriter(const RequestWriter&) = delete;
        RequestWriter& operator=(const RequestWriter&) = delete;

    private:
        std::string request_shm_name_;
        std::string bulk_shm_name_;

        int request_shm_fd_ = -1;
        void* request_shm_map_ptr_ = nullptr;
        RequestSlot* request_slots_ = nullptr;
        RequestQueueControl* request_queue_control_ = nullptr;

        int bulk_shm_fd_ = -1;
        void* bulk_shm_map_ptr_ = nullptr;
        size_t bulk_shm_current_offset_ = 0;

        int kernel_event_fd_ = -1;
        uintptr_t kqueue_ident_ = 1;

        bool initialize_ipc_resources();
        void cleanup_ipc_resources();
        void trigger_kernel_event();

        uint64_t write_prompt_to_bulk_shm(const std::string& prompt_string);
        std::unique_ptr<SharedMemoryManager> bulk_shm_manager_;
    };

    RequestWriter* get_global_request_writer();
    void init_global_request_writer();
    void shutdown_global_request_writer();

} // namespace pie_core::ipc
