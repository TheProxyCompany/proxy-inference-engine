#pragma once

#include <string>
#include <stdexcept>

namespace pie_core::ipc {

    class IPCError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class IPCManager {
    public:
        /**
         * @brief Creates the necessary IPC resources (SHM segments, kernel event).
         * @param request_shm_name Name for the request queue SHM segment.
         * @param response_shm_name Name for the response queue SHM segment.
         * @throws IPCError if resource creation fails.
         */
        IPCManager(const std::string& request_shm_name, const std::string& response_shm_name);

        /**
         * @brief Destroys (unlinks) the created IPC resources.
         */
        ~IPCManager();

        /**
         * @brief Gets the file descriptor for kernel event notification.
         *        On macOS, this is the kqueue fd.
         *        On Linux, this is the eventfd.
         * @return The kernel event file descriptor.
         */
        [[nodiscard]] int get_kernel_event_fd() const noexcept;

        /**
         * @brief Manually triggers the kernel event to wake up waiters.
         */
        void trigger_kernel_event();

        // Prevent copying/moving
        IPCManager(const IPCManager&) = delete;
        IPCManager& operator=(const IPCManager&) = delete;
        IPCManager(IPCManager&&) = delete;
        IPCManager& operator=(IPCManager&&) = delete;

    private:
        std::string request_shm_name_;
        std::string response_shm_name_;
        int kernel_event_fd_ = -1;
        uintptr_t kqueue_ident_ = 1; // Used only on macOS for user events

        bool create_shm_segment(const std::string& name, size_t size);
        bool initialize_kernel_event();
        void cleanup_shm_segment(const std::string& name);
        void cleanup_kernel_event();
    };

} // namespace pie_core::ipc
