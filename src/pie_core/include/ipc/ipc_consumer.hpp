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
            const std::string& response_shm_name = RESPONSE_QUEUE_SHM_NAME, // From ipc_response.hpp
            const std::string& bulk_response_shm_name = "/pie_bulk_response_data" // If using bulk for large non-streaming
        );
        ~IPCConsumer();

        // Blocking call to wait for and retrieve the next available response delta.
        // Returns std::nullopt on timeout or shutdown.
        // The Python side would call this repeatedly in a background thread.
        std::optional<ResponseDeltaSlot> consume_next_delta(int timeout_ms = -1); // -1 for blocking

        // For non-streaming, if full responses are in a bulk SHM
        // std::optional<std::vector<char>> consume_bulk_response(uint64_t offset, uint64_t size);

        IPCConsumer(const IPCConsumer&) = delete;
        IPCConsumer& operator=(const IPCConsumer&) = delete;

    private:
        std::string response_shm_name_;
        std::string bulk_response_shm_name_; // If used

        int response_shm_fd_ = -1;
        void* response_shm_map_ptr_ = nullptr;
        ResponseDeltaSlot* response_slots_ = nullptr;
        ResponseQueueControl* response_queue_control_ = nullptr; // Similar to request side

        // Bulk SHM for large full responses (if applicable)
        // int bulk_response_shm_fd_ = -1;
        // void* bulk_response_shm_map_ptr_ = nullptr;


        int kernel_event_fd_ = -1; // FD for Python to wait on (C++ engine triggers it)

        bool initialize_ipc_resources();
        void cleanup_ipc_resources();
        bool wait_for_event(int timeout_ms);
    };

    // Global/static management for nanobind access from Python process
    IPCConsumer* get_global_ipc_consumer();
    void init_global_ipc_consumer();
    void shutdown_global_ipc_consumer();

} // namespace pie_core::ipc
