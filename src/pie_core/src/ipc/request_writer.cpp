#include "ipc/request_writer.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <spdlog/spdlog.h>
#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/eventfd.h>
#endif

namespace pie_core::ipc {

    // --- Global Instance Management ---
    std::unique_ptr<RequestWriter> global_producer_instance = nullptr;

    RequestWriter* get_global_request_writer() {
        if (!global_producer_instance) {
            throw std::runtime_error("RequestWriter global instance not initialized. Call init_global_request_writer() from Python first.");
        }
        return global_producer_instance.get();
    }


    void init_global_request_writer() {
        if (!global_producer_instance) {
            try {
                global_producer_instance = std::make_unique<RequestWriter>();
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed to initialize global RequestWriter: ") + e.what());
            }
        }
    }

    void shutdown_global_request_writer() {
        global_producer_instance.reset();
    }


    RequestWriter::RequestWriter(const std::string& request_shm_name, const std::string& bulk_shm_name) :
        request_shm_name_(request_shm_name),
        bulk_shm_name_(bulk_shm_name)
    {
        if (!initialize_ipc_resources()) {
            throw std::runtime_error("RequestWriter: Failed to initialize IPC resources.");
        }
        #if defined(__APPLE__)
            kernel_event_fd_ = -1;
            kqueue_ident_ = 1;
            spdlog::warn("RequestWriter: kqueue event triggering from a separate process needs a robust mechanism (e.g., nanobind call to engine's kqueue trigger function or named pipe).");
        #elif defined(__linux__)
            kernel_event_fd_ = -1; // Placeholder
            spdlog::warn("RequestWriter: eventfd mechanism needs the engine's eventfd to be accessible by the producer process.");
        #else
            #error "Kernel event notification not implemented for this platform."
        #endif
    }

    RequestWriter::~RequestWriter() {
        cleanup_ipc_resources();
        if (kernel_event_fd_ != -1 && (
            #if defined(__linux__) // Only close if it's an FD we opened
            true
            #else
            false // For kqueue, this producer doesn't own the engine's kqueue fd
            #endif
        )) {
            close(kernel_event_fd_);
        }
        spdlog::info("RequestWriter destroyed.");
    }

    bool RequestWriter::initialize_ipc_resources() {
        spdlog::info("RequestWriter: Initializing IPC resources...");

        // 1. Initialize Request Queue SHM (Open Existing)
        request_shm_fd_ = shm_open(request_shm_name_.c_str(), O_RDWR, 0);
        if (request_shm_fd_ == -1) {
            spdlog::error("RequestWriter: shm_open for request queue '{}' failed: {}", request_shm_name_, strerror(errno));
            return false;
        }
        spdlog::info("RequestWriter: Opened request SHM '{}', fd={}", request_shm_name_, request_shm_fd_);

        // 2. Map Request Queue SHM
        request_shm_map_ptr_ = mmap(nullptr, REQUEST_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, request_shm_fd_, 0);
        if (request_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("RequestWriter: mmap for request queue '{}' failed: {}", request_shm_name_, strerror(errno));
            close(request_shm_fd_); request_shm_fd_ = -1;
            return false;
        }
        spdlog::debug("RequestWriter: Mapped request SHM '{}' at address {:p}", request_shm_name_, request_shm_map_ptr_);

        // 3. Initialize Request Queue Control
        // Control block is at the beginning of the shared memory
        request_queue_control_ = static_cast<RequestQueueControl*>(request_shm_map_ptr_);
        // Slots start after the control block
        request_slots_ = reinterpret_cast<RequestSlot*>(static_cast<char*>(request_shm_map_ptr_) + sizeof(RequestQueueControl));
        spdlog::debug("RequestWriter: Set up request_queue_control_ at {:p}, request_slots_ at {:p}",
                     (void*)request_queue_control_, (void*)request_slots_);

        // 4. Initialize Bulk Data SHM (Open Existing)
        try {
            bulk_shm_manager_ = std::make_unique<SharedMemoryManager>(
                bulk_shm_name_,
                0,
                false
            );
            bulk_shm_map_ptr_ = bulk_shm_manager_->get_segment_base_address();
            spdlog::info("RequestWriter: SharedMemoryManager for bulk data SHM '{}' initialized.", bulk_shm_name_);
        } catch (const std::exception& e) {
            spdlog::error("RequestWriter: Error initializing SharedMemoryManager'{}': {}", bulk_shm_name_, e.what());
            if (request_shm_map_ptr_ != MAP_FAILED) munmap(request_shm_map_ptr_, REQUEST_QUEUE_SHM_SIZE);
            if (request_shm_fd_ != -1) close(request_shm_fd_);
            request_shm_map_ptr_ = nullptr; request_shm_fd_ = -1;
            return false;
        }

        spdlog::info("RequestWriter: IPC resources initialized successfully.");
        return true;
    }

    void RequestWriter::cleanup_ipc_resources() {
        spdlog::info("RequestWriter: Cleaning up IPC resources...");

        bulk_shm_manager_.reset();

        // Cleanup Request Queue SHM
        if (request_shm_map_ptr_ != nullptr && request_shm_map_ptr_ != MAP_FAILED) {
            if (munmap(request_shm_map_ptr_, REQUEST_QUEUE_SHM_SIZE) == -1) {
                spdlog::error("RequestWriter: munmap for request queue '{}' failed: {}", request_shm_name_, strerror(errno));
            }
            request_shm_map_ptr_ = nullptr;
        }
        if (request_shm_fd_ != -1) {
            if (close(request_shm_fd_) == -1) {
                spdlog::error("RequestWriter: close for request queue fd {} failed: {}", request_shm_fd_, strerror(errno));
            }
            request_shm_fd_ = -1;
        }
        request_queue_control_ = nullptr;
        request_slots_ = nullptr;
        spdlog::info("RequestWriter: IPC resources cleaned up.");
    }

    uint64_t RequestWriter::write_prompt_to_bulk_shm(const std::string& prompt_string) {
        if (!bulk_shm_manager_) {
            throw std::runtime_error("RequestWriter: Bulk SHM manager not initialized for writing prompt.");
        }

        size_t data_size = prompt_string.length();
        if (data_size == 0) {
            spdlog::warn("RequestWriter: Attempting to write an empty prompt string to bulk SHM.");
            data_size = 1; // Allocate 1 byte for empty string to get a valid ptr
        }

        void* shm_block_ptr = nullptr;
        try {
            shm_block_ptr = bulk_shm_manager_->allocate(data_size);
        } catch (const SharedMemoryError& e) {
            spdlog::error("RequestWriter: Failed to allocate {} bytes in bulk SHM for prompt: {}", data_size, e.what());
            throw; // Re-throw
        }

        if (!shm_block_ptr) {
            throw std::runtime_error("RequestWriter: Bulk SHM allocation returned nullptr for prompt.");
        }

        if (!prompt_string.empty()) {
            std::memcpy(shm_block_ptr, prompt_string.data(), prompt_string.length());
        } else if (data_size == 1) {
            static_cast<char*>(shm_block_ptr)[0] = '\0';
        }

        void* bulk_shm_base = bulk_shm_manager_->get_segment_base_address();
        if (!bulk_shm_base) {
            bulk_shm_manager_->deallocate(shm_block_ptr);
            throw std::runtime_error("RequestWriter: Could not get bulk SHM base address for offset calculation.");
        }
        uint64_t offset = static_cast<char*>(shm_block_ptr) - static_cast<char*>(bulk_shm_base);

        spdlog::debug("RequestWriter: Wrote prompt of size {} to bulk SHM at offset {}", prompt_string.length(), offset);
        return offset;
    }

    void RequestWriter::trigger_kernel_event() {
        #if defined(__APPLE__)
            struct kevent change;
            EV_SET(&change, kqueue_ident_, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr); // Trigger user event
            if (kevent(kernel_event_fd_, &change, 1, nullptr, 0, nullptr) == -1) {
                perror("RequestWriter: kevent trigger failed");
            }
        #elif defined(__linux__)
            uint64_t u = 1;
            if (write(kernel_event_fd_, &u, sizeof(uint64_t)) != sizeof(uint64_t)) {
                perror("RequestWriter: eventfd write failed");
            }
        #endif
    }

    uint64_t RequestWriter::submit_request_to_engine(
        uint64_t request_id,
        const std::string& prompt_string,
        const sequence::SamplingParams& sampling_params,
        const sequence::LogitsParams& logits_params,
        const sequence::StopCriteria& stop_criteria,
        const sequence::IPCHandles& ipc_handles,
        const std::string& tool_schemas_str,
        const std::string& response_format_str
    ) {
        if (!request_queue_control_ || !request_slots_) {
            throw std::runtime_error("RequestWriter: SHM for requests not initialized.");
        }
        if (!bulk_shm_map_ptr_) {
            throw std::runtime_error("RequestWriter: Bulk SHM not initialized.");
        }

        // 1. Write prompt to bulk SHM
        uint64_t prompt_offset = write_prompt_to_bulk_shm(prompt_string);
        uint64_t prompt_size = prompt_string.length();

        // 2. Claim a slot in RequestQueue
        uint64_t producer_ticket = request_queue_control_->producer_idx.fetch_add(1, std::memory_order_acq_rel);
        uint64_t slot_idx = producer_ticket % REQUEST_QUEUE_NUM_SLOTS;
        RequestSlot& slot = request_slots_[slot_idx];

        // 3. Wait for a slot to become FREE in the RequestQueue
        RequestState expected_free = RequestState::FREE;
        int spin_count = 0;
        const int max_spins = 1000000; // Adjust as needed, roughly 1 second if sleep is 1us
        while (!slot.state.compare_exchange_weak(expected_free, RequestState::WRITING,
                                                 std::memory_order_acq_rel, std::memory_order_relaxed)) {
            expected_free = RequestState::FREE; // Reset for next attempt
            spin_count++;
            if (spin_count > max_spins) {
                throw std::runtime_error("RequestWriter: Timeout waiting for a free request slot. Engine might be stuck or queue full.");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        // 4. Fill RequestSlot
        slot.request_id = request_id;
        slot.prompt_shm_offset = prompt_offset;
        slot.prompt_shm_size = prompt_size;
        slot.sampling_params = sampling_params;
        slot.logits_params = logits_params;
        slot.stop_criteria = stop_criteria;
        slot.ipc_handles = ipc_handles;
        slot.tool_schemas_str = tool_schemas_str;
        slot.response_format_str = response_format_str;
        std::atomic_thread_fence(std::memory_order_release);

        // 5. Mark READY
        slot.state.store(RequestState::READY, std::memory_order_release);

        // // 6. Trigger kernel event
        // trigger_kernel_event();

        return request_id;
    }

} // namespace pie_core::ipc
