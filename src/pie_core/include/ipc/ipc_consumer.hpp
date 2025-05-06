#pragma once

#include "ipc_response.hpp"
#include <string>
#include <optional>
#include <cstdint>
#include <memory>

namespace pie_core::ipc {

    struct ResponseDeltaSlot;

    class IPCConsumer {
    public:
        IPCConsumer(
            const std::string& response_shm_name = RESPONSE_QUEUE_SHM_NAME,
            const std::string& bulk_response_shm_name = "/pie_bulk_response_data"
        );
        ~IPCConsumer();

        std::optional<ResponseDeltaSlot> consume_next_delta(int timeout_ms = -1);

        IPCConsumer(const IPCConsumer&) = delete;
        IPCConsumer& operator=(const IPCConsumer&) = delete;

    private:
        std::string response_shm_name_;

        int response_shm_fd_ = -1;
        void* response_shm_map_ptr_ = nullptr;
        ResponseDeltaSlot* response_slots_ = nullptr;
        ResponseQueueControl* response_queue_control_ = nullptr;

        int kernel_event_fd_ = -1;

        bool initialize_ipc_resources();
        void cleanup_ipc_resources();
        bool wait_for_event(int timeout_ms);
    };

    IPCConsumer* get_global_ipc_consumer();
    void init_global_ipc_consumer();
    void shutdown_global_ipc_consumer();

} // namespace pie_core::ipc
