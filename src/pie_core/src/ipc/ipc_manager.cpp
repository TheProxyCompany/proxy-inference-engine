#include "ipc/ipc_manager.hpp"
#include "ipc/request.hpp"
#include "ipc/response.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <spdlog/spdlog.h>
#include <atomic>

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/eventfd.h>
#else
#error "Unsupported platform for IPCManager kernel events"
#endif

namespace pie_core::ipc {

    IPCManager::IPCManager(
        const std::string& request_shm_name,
        const std::string& response_shm_name
    )
    : request_shm_name_(request_shm_name),
      response_shm_name_(response_shm_name)
    {
        spdlog::info("IPCManager: Initializing...");
        bool success = true;

        // Create Request Queue SHM
        success &= create_shm_segment(request_shm_name_, REQUEST_QUEUE_SHM_SIZE);
        if (!success) {
            throw IPCError("Failed to create request queue SHM segment: " + request_shm_name_);
        }

        // Create Response Queue SHM
        success &= create_shm_segment(response_shm_name_, RESPONSE_QUEUE_SHM_SIZE);
        if (!success) {
            cleanup_shm_segment(request_shm_name_); // Clean up partially created resources
            throw IPCError("Failed to create response queue SHM segment: " + response_shm_name_);
        }

        // Initialize Kernel Event Mechanism
        success &= initialize_kernel_event();
        if (!success) {
            cleanup_shm_segment(request_shm_name_);
            cleanup_shm_segment(response_shm_name_);
            throw IPCError("Failed to initialize kernel event mechanism.");
        }

        spdlog::info("IPCManager: Initialization successful.");
    }

    IPCManager::~IPCManager() {
        spdlog::info("IPCManager: Cleaning up...");
        cleanup_kernel_event();
        cleanup_shm_segment(response_shm_name_);
        cleanup_shm_segment(request_shm_name_);
        spdlog::info("IPCManager: Cleanup complete.");
    }

    bool IPCManager::create_shm_segment(const std::string& name, size_t size) {
        // 1. Create/Open SHM
        int shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            spdlog::error("IPCManager: shm_open failed for '{}': {}", name, strerror(errno));
            return false;
        }

        // 2. Set Size
        if (ftruncate(shm_fd, size) == -1) {
            spdlog::error("IPCManager: ftruncate failed for '{}' (size {}): {}", name, size, strerror(errno));
            close(shm_fd);
            shm_unlink(name.c_str()); // Clean up on failure
            return false;
        }

        // 3. Map temporarily to initialize control block (atomics)
        void* map_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (map_ptr == MAP_FAILED) {
            spdlog::error("IPCManager: mmap failed for initialization of '{}': {}", name, strerror(errno));
            close(shm_fd);
            shm_unlink(name.c_str());
            return false;
        }

        // Zero out the memory, especially the control block
        std::memset(map_ptr, 0, size);

        // Specifically initialize atomic indices to 0 if it's a known queue type
        if (name == REQUEST_QUEUE_SHM_NAME) {
             RequestQueueControl* control = static_cast<RequestQueueControl*>(map_ptr);
             control->producer_idx.store(0, std::memory_order_relaxed);
             control->consumer_idx.store(0, std::memory_order_relaxed);
             // Initialize slot states to FREE
             RequestSlot* slots = reinterpret_cast<RequestSlot*>(static_cast<char*>(map_ptr) + sizeof(RequestQueueControl));
             for (size_t i = 0; i < REQUEST_QUEUE_NUM_SLOTS; ++i) {
                 slots[i].state.store(RequestState::FREE, std::memory_order_relaxed);
             }
             spdlog::info("IPCManager: Initialized RequestQueueControl for '{}'", name);
        } else if (name == RESPONSE_QUEUE_SHM_NAME) {
            ResponseQueueControl* control = static_cast<ResponseQueueControl*>(map_ptr);
             control->producer_idx.store(0, std::memory_order_relaxed);
             control->consumer_idx.store(0, std::memory_order_relaxed);
             // Initialize slot states to FREE_FOR_CPP_WRITER
             ResponseDeltaSlot* slots = reinterpret_cast<ResponseDeltaSlot*>(static_cast<char*>(map_ptr) + sizeof(ResponseQueueControl));
             for (size_t i = 0; i < RESPONSE_QUEUE_NUM_SLOTS; ++i) {
                 slots[i].state.store(ResponseSlotState::FREE_FOR_CPP_WRITER, std::memory_order_relaxed);
             }
             spdlog::info("IPCManager: Initialized ResponseQueueControl for '{}'", name);
        }


        // 4. Unmap and Close FD (Reader/Writer will open/map their own)
        if (munmap(map_ptr, size) == -1) {
            spdlog::error("IPCManager: munmap failed after initialization of '{}': {}", name, strerror(errno));
            // Continue cleanup despite munmap error
        }
        if (close(shm_fd) == -1) {
            spdlog::error("IPCManager: close failed after initialization of '{}': {}", name, strerror(errno));
            // Continue cleanup despite close error
        }

        spdlog::info("IPCManager: SHM segment '{}' created and initialized.", name);
        return true;
    }

    bool IPCManager::initialize_kernel_event() {
        #if defined(__APPLE__)
            kernel_event_fd_ = kqueue();
            if (kernel_event_fd_ == -1) {
                spdlog::error("IPCManager: kqueue() failed: {}", strerror(errno));
                return false;
            }
            // Register the user event source that the writer will trigger
            struct kevent change;
            EV_SET(&change, kqueue_ident_, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
            if (kevent(kernel_event_fd_, &change, 1, nullptr, 0, nullptr) == -1) {
                 spdlog::error("IPCManager: kevent() failed to add user event: {}", strerror(errno));
                 close(kernel_event_fd_);
                 kernel_event_fd_ = -1;
                 return false;
            }
            spdlog::info("IPCManager: kqueue initialized (fd={}).", kernel_event_fd_);
        #elif defined(__linux__)
            kernel_event_fd_ = eventfd(0, EFD_SEMAPHORE); // Use semaphore behavior
            if (kernel_event_fd_ == -1) {
                spdlog::error("IPCManager: eventfd() failed: {}", strerror(errno));
                return false;
            }
            spdlog::info("IPCManager: eventfd initialized (fd={}).", kernel_event_fd_);
        #endif
        return true;
    }

    void IPCManager::cleanup_shm_segment(const std::string& name) {
        if (shm_unlink(name.c_str()) == -1) {
            // This might fail if the segment was already removed or never created properly
            // Log as warning unless it's an unexpected error like EACCES
            if (errno != ENOENT) {
                spdlog::warn("IPCManager: shm_unlink failed for '{}': {} (Segment might already be removed)", name, strerror(errno));
            }
        } else {
            spdlog::info("IPCManager: SHM segment '{}' unlinked.", name);
        }
    }

    void IPCManager::cleanup_kernel_event() {
        if (kernel_event_fd_ != -1) {
            if (close(kernel_event_fd_) == -1) {
                spdlog::error("IPCManager: close failed for kernel event fd {}: {}", kernel_event_fd_, strerror(errno));
            } else {
                spdlog::info("IPCManager: Kernel event fd {} closed.", kernel_event_fd_);
            }
            kernel_event_fd_ = -1;
        }
    }

    int IPCManager::get_kernel_event_fd() const noexcept {
        return kernel_event_fd_;
    }

} // namespace pie_core::ipc
