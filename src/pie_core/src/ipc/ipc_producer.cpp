#include "ipc/ipc_producer.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/eventfd.h>
#endif

namespace pie_core::ipc {

    // --- Global Instance Management (Example) ---
    std::unique_ptr<IPCProducer> global_producer_instance = nullptr;

    IPCProducer* get_global_ipc_producer() {
        if (!global_producer_instance) {
            init_global_ipc_producer();
        }
        return global_producer_instance.get();
    }

    void init_global_ipc_producer() {
        if (!global_producer_instance) {
            global_producer_instance = std::make_unique<IPCProducer>();
        }
    }

    void shutdown_global_ipc_producer() {
        global_producer_instance.reset();
    }


    IPCProducer::IPCProducer(const std::string& request_shm_name, const std::string& bulk_shm_name) {
        this->request_shm_name_ = request_shm_name;
        this->bulk_shm_name_ = bulk_shm_name;
        if (!initialize_ipc_resources()) {
            throw std::runtime_error("IPCProducer: Failed to initialize IPC resources.");
        }
        #if defined(__APPLE__)
            kernel_event_fd_ = kqueue();
            if (kernel_event_fd_ == -1) {
                perror("IPCProducer: kqueue() failed");
                cleanup_ipc_resources();
                throw std::runtime_error("IPCProducer: kqueue creation failed.");
            }
        #elif defined(__linux__)
            kernel_event_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
            if (kernel_event_fd_ == -1) {
                perror("IPCProducer: eventfd() failed");
                cleanup_ipc_resources();
                throw std::runtime_error("IPCProducer: eventfd creation failed.");
            }
        #else
            #error "Kernel event notification not implemented for this platform."
        #endif
    }

    IPCProducer::~IPCProducer() {
        cleanup_ipc_resources();
        if (kernel_event_fd_ != -1) {
            close(kernel_event_fd_);
        }
    }

    bool IPCProducer::initialize_ipc_resources() {
        // TODO: Implement
        return false;
    }

    void IPCProducer::cleanup_ipc_resources() {
        // TODO: Implement
    }

    uint64_t IPCProducer::write_prompt_to_bulk_shm(const std::vector<int32_t>& prompt_tokens) {
        // TODO: Implement
        return 0;
    }

    void IPCProducer::trigger_kernel_event() {
        #if defined(__APPLE__)
            struct kevent change;
            EV_SET(&change, kqueue_ident_, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr); // Trigger user event
            if (kevent(kernel_event_fd_, &change, 1, nullptr, 0, nullptr) == -1) {
                perror("IPCProducer: kevent trigger failed");
            }
        #elif defined(__linux__)
            uint64_t u = 1;
            if (write(kernel_event_fd_, &u, sizeof(uint64_t)) != sizeof(uint64_t)) {
                perror("IPCProducer: eventfd write failed");
            }
        #endif
    }

    uint64_t IPCProducer::submit_request_to_engine(
        uint64_t request_id,
        const std::vector<int32_t>& prompt_tokens,
        const sequence::SamplingParams& sampling_params,
        const sequence::LogitsParams& logits_params,
        const sequence::StopCriteria& stop_criteria
    ) {
        if (!request_queue_control_ || !request_slots_) {
            throw std::runtime_error("IPCProducer: SHM for requests not initialized.");
        }
        if (!bulk_shm_map_ptr_) {
            throw std::runtime_error("IPCProducer: Bulk SHM not initialized.");
        }

        // 1. Write prompt to bulk SHM
        uint64_t prompt_offset = write_prompt_to_bulk_shm(prompt_tokens);
        uint64_t prompt_size = prompt_tokens.size() * sizeof(int32_t);

        // 2. Claim a slot in RequestQueue
        uint64_t producer_ticket = request_queue_control_->producer_idx.fetch_add(1, std::memory_order_acq_rel);
        uint64_t slot_idx = producer_ticket % REQUEST_QUEUE_NUM_SLOTS;
        RequestSlot& slot = request_slots_[slot_idx];

        // Spin-wait for the slot to become FREE (C++ engine should set it to FREE after reading)
        // Add a timeout to prevent indefinite spinning
        RequestState expected_free = RequestState::FREE;
        int spin_count = 0;
        const int max_spins = 1000000; // Adjust as needed, roughly 1 second if sleep is 1us
        while (!slot.state.compare_exchange_weak(expected_free, RequestState::WRITING,
                                                 std::memory_order_acq_rel, std::memory_order_relaxed)) {
            expected_free = RequestState::FREE; // Reset for next attempt
            spin_count++;
            if (spin_count > max_spins) {
                 // Failed to acquire slot, revert producer_idx (this is tricky and can lead to issues)
                 // Or better, signal an error back to Python.
                 // For now, throw.
                throw std::runtime_error("IPCProducer: Timeout waiting for a free request slot. Engine might be stuck or queue full.");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1)); // Prevent hard spin
        }
        // Successfully CASed from FREE to WRITING

        // 3. Fill RequestSlot
        slot.request_id = request_id;
        slot.prompt_shm_offset = prompt_offset;
        slot.prompt_shm_size = prompt_size;
        slot.sampling_params = sampling_params;
        slot.logits_params = logits_params;
        slot.stop_criteria = stop_criteria;
        // slot.ipc_handles = ...; // Python needs to provide info for response path

        // 4. Mark READY
        slot.state.store(RequestState::READY, std::memory_order_release);

        // 5. Trigger kernel event
        trigger_kernel_event();

        return request_id;
    }

} // namespace pie_core::ipc
