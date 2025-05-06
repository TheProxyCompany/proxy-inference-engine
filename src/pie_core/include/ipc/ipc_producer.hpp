#pragma once

#include "ipc_request.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"

#include <string>
#include <vector>
#include <cstdint>

namespace pie_core::ipc {

    constexpr const char* BULK_DATA_SHM_NAME = "/pie_bulk_data";
    constexpr size_t BULK_DATA_SHM_SIZE = 1024 * 1024 * 256; // 256MB

    class IPCProducer {
    public:
        IPCProducer(
            const std::string& request_shm_name = REQUEST_QUEUE_SHM_NAME,
            const std::string& bulk_shm_name = BULK_DATA_SHM_NAME
        );
        ~IPCProducer();

        uint64_t submit_request_to_engine(
            uint64_t request_id,
            const std::vector<int32_t>& prompt_tokens,
            const sequence::SamplingParams& sampling_params,
            const sequence::LogitsParams& logits_params,
            const sequence::StopCriteria& stop_criteria
        );

        IPCProducer(const IPCProducer&) = delete;
        IPCProducer& operator=(const IPCProducer&) = delete;

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

        uint64_t write_prompt_to_bulk_shm(const std::vector<int32_t>& prompt_tokens);
    };

    IPCProducer* get_global_ipc_producer();
    void init_global_ipc_producer();
    void shutdown_global_ipc_producer();

} // namespace pie_core::ipc
