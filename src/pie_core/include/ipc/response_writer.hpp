#pragma once

#include "ipc/response.hpp"
#include "sequence/sequence.hpp"
#include <string>
#include <memory>
#include <atomic>
#include <cstdint>
#include <vector>

namespace pie_core::ipc {

    class ResponseWriterError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    /**
     * @brief Writes generated token deltas and final status to the response shared memory queue.
     * This class runs within the C++ Engine process.
     */
    class ResponseWriter {
    public:
        /**
         * @brief Constructor. Opens the existing response queue SHM segment.
         * @param response_shm_name The name of the response shared memory segment.
         * @throws ResponseWriterError if initialization fails.
         */
        explicit ResponseWriter(const std::string& response_shm_name = RESPONSE_QUEUE_SHM_NAME);

        /**
         * @brief Destructor. Cleans up mapped memory and file descriptor.
         */
        ~ResponseWriter();

        /**
         * @brief Writes a single response delta (containing one or more tokens) to the queue.
         * @param delta The data to write into a response slot.
         * @throws ResponseWriterError if writing fails (e.g., queue full timeout).
         */
        void write_delta(const ResponseDeltaSlot& delta);

        // --- Prevent Copying/Moving ---
        ResponseWriter(const ResponseWriter&) = delete;
        ResponseWriter& operator=(const ResponseWriter&) = delete;
        ResponseWriter(ResponseWriter&&) = delete;
        ResponseWriter& operator=(ResponseWriter&&) = delete;

    private:
        std::string response_shm_name_;
        int response_shm_fd_ = -1;
        void* response_shm_map_ptr_ = nullptr;
        ResponseDeltaSlot* response_slots_ = nullptr;
        ResponseQueueControl* response_queue_control_ = nullptr;

        bool initialize_ipc_resources();
        void cleanup_ipc_resources();
    };

} // namespace pie_core::ipc
