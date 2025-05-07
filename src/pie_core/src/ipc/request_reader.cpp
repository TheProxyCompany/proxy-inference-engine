#include "ipc/request_reader.hpp"
#include "engine/raw_request.hpp"

#include <sys/event.h> // Ensure this is included for timespec
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <time.h>      // For timespec
#include <chrono>

namespace pie_core::ipc {

    RequestReader::RequestReader(
        RawRequestQueue& output_queue,
        SharedMemoryManager& shm_manager,
        const std::string& request_shm_name,
        int kernel_event_fd
    ):
        output_queue_(output_queue),
        kernel_event_fd_(kernel_event_fd),
        shm_manager_(shm_manager)
    {
        spdlog::info("RequestReader: Initializing with request_shm_name='{}', kernel_event_fd={}",
                   request_shm_name, kernel_event_fd);

        if (!initialize_ipc_resources(request_shm_name)) {
            spdlog::critical("RequestReader: Failed to initialize IPC resources for '{}'", request_shm_name);
            throw std::runtime_error("RequestReader: Failed to initialize IPC resources");
        }

        spdlog::info("RequestReader: Successfully initialized");
    }

    RequestReader::~RequestReader() {
        spdlog::info("RequestReader: Destructor called");
        stop();
        cleanup_ipc_resources();
        spdlog::debug("RequestReader: Destructor complete, all resources cleaned up");
    }

    void RequestReader::run_loop() {
        spdlog::info("RequestReader: Run loop entered");
        uint64_t loop_counter = 0;
        uint64_t requests_processed = 0;

        while (!stop_flag_.load(std::memory_order_acquire)) {
            loop_counter++;

            // Log occasional statistics at trace level
            if (loop_counter % 1000 == 0) {
                spdlog::trace("RequestReader: Run loop iteration {}, processed {} requests so far",
                             loop_counter, requests_processed);
            }

            // Wait for notification OR timeout
            bool event_received = wait_for_notification();

            // Check stop flag *after* waiting/polling
            if (stop_flag_.load(std::memory_order_acquire)) {
                spdlog::debug("RequestReader: Stop flag detected after wait, exiting loop");
                break;
            }

            // Always try processing, even on timeout, as polling might find data
            // written between polls without an event trigger succeeding.
            process_incoming_requests();

            // If no event was received (timeout), add a minimal sleep
            // to prevent extremely tight spin if queue remains empty.
            // This is less critical now since kevent has a timeout, but doesn't hurt.
            if (!event_received) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        spdlog::info("RequestReader: Run loop exited after {} iterations, processed {} total requests",
                   loop_counter, requests_processed);
    }

    void RequestReader::stop() {
        bool was_already_stopping = stop_flag_.exchange(true, std::memory_order_acq_rel);
        if (!was_already_stopping) {
            spdlog::info("RequestReader: Stop signal received");
        } else {
            spdlog::debug("RequestReader: Duplicate stop signal received (already stopping)");
        }
    }

    bool RequestReader::initialize_ipc_resources(const std::string& name) {
        spdlog::debug("RequestReader: Initializing IPC resources for '{}'", name);

        // 1. Open SHM
        request_shm_fd_ = shm_open(
            name.c_str(),
            O_RDWR,
            0
        );
        if (request_shm_fd_ < 0) {
            spdlog::error("RequestReader: shm_open failed for '{}': {} (errno={})",
                         name, strerror(errno), errno);
            return false;
        }
        spdlog::debug("RequestReader: Opened SHM '{}', fd={}", name, request_shm_fd_);

        // 2. Map SHM
        request_shm_map_ptr_ = mmap(
            nullptr,
            REQUEST_QUEUE_SHM_SIZE,
            PROT_READ|PROT_WRITE,
            MAP_SHARED,
            request_shm_fd_,
            0
        );
        if (request_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("RequestReader: mmap failed for '{}': {} (errno={})",
                          name, strerror(errno), errno);
            close(request_shm_fd_);
            request_shm_fd_ = -1;
            return false;
        }
        spdlog::debug("RequestReader: Mapped SHM '{}' at address {:p}", name, request_shm_map_ptr_);

        // 3. Set up pointers to control structure and slots
        // Control block is at the beginning of the shared memory
        request_queue_control_ = static_cast<RequestQueueControl*>(request_shm_map_ptr_);
        // Slots start after the control block
        request_slots_ = reinterpret_cast<RequestSlot*>(
            static_cast<char*>(request_shm_map_ptr_) + sizeof(RequestQueueControl)
        );
        spdlog::debug("RequestReader: Set up request_queue_control_ at {:p}, request_slots_ at {:p}",
                     (void*)request_queue_control_, (void*)request_slots_);

        // 4. Get bulk data base address
        bulk_data_map_ptr_ = shm_manager_.get_segment_base_address();
        spdlog::debug("RequestReader: Got bulk data segment base address: {:p}", bulk_data_map_ptr_);

        // 5. Read initial producer/consumer indices
        uint64_t initial_prod_idx = request_queue_control_->producer_idx.load(std::memory_order_acquire);
        uint64_t initial_cons_idx = request_queue_control_->consumer_idx.load(std::memory_order_acquire);
        spdlog::info("RequestReader: Initial queue state - producer_idx={}, consumer_idx={}",
                    initial_prod_idx, initial_cons_idx);

        return true;
    }

    void RequestReader::cleanup_ipc_resources() {
        spdlog::debug("RequestReader: Cleaning up IPC resources");

        if (request_shm_map_ptr_ && request_shm_map_ptr_ != MAP_FAILED) {
            spdlog::debug("RequestReader: Unmapping request SHM at {:p}", request_shm_map_ptr_);
            if (munmap(request_shm_map_ptr_, REQUEST_QUEUE_SHM_SIZE) == -1) {
                spdlog::error("RequestReader: munmap failed: {}", strerror(errno));
            }
            request_shm_map_ptr_ = nullptr;
            request_slots_ = nullptr;
            request_queue_control_ = nullptr;
        }

        if (request_shm_fd_ >= 0) {
            spdlog::debug("RequestReader: Closing request SHM fd {}", request_shm_fd_);
            if (close(request_shm_fd_) == -1) {
                spdlog::error("RequestReader: close failed for fd {}: {}", request_shm_fd_, strerror(errno));
            }
            request_shm_fd_ = -1;
        }

        spdlog::debug("RequestReader: IPC resources cleanup complete");
    }

    bool RequestReader::wait_for_notification() {
        if (kernel_event_fd_ < 0) {
            spdlog::error("RequestReader: Invalid kernel event fd ({}) in wait_for_notification", kernel_event_fd_);
            // Fallback to short sleep to prevent tight spin loop if fd is bad
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return false; // Indicate potential error or just timeout
        }

        struct kevent kev_in;
        // Wait for the user event IDENT we registered in IPCManager
        EV_SET(&kev_in, kqueue_ident_, EVFILT_USER, EV_ADD | EV_ENABLE | EV_CLEAR, 0, 0, nullptr);

        // Set a short timeout (e.g., 10 milliseconds)
        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = 10 * 1000000; // 10 milliseconds

        struct kevent kev_out;
        // Pass the timeout to kevent
        int nevents = kevent(kernel_event_fd_, &kev_in, 1, &kev_out, 1, &timeout);

        if (nevents == -1) {
            spdlog::error("RequestReader: kevent wait failed: {} (errno={})", strerror(errno), errno);
            return false; // Error occurred
        } else if (nevents == 0) {
            // Timeout occurred - this is EXPECTED when polling and no event arrived
            spdlog::trace("RequestReader: kevent timed out (polling)");
            return false; // Indicate timeout, loop will continue and check queue state
        } else {
            // Event received
            if (kev_out.filter == EVFILT_USER && kev_out.ident == kqueue_ident_) {
                spdlog::trace("RequestReader: Received kernel event notification (ident={}, filter={})",
                             kev_out.ident, kev_out.filter);
                return true; // Event successfully received
            } else {
                spdlog::warn("RequestReader: Received unexpected kernel event (ident={}, filter={})",
                             kev_out.ident, kev_out.filter);
                return false; // Unexpected event, treat as timeout/error for now
            }
        }
    }

    void RequestReader::process_incoming_requests() {
        auto prod = request_queue_control_->producer_idx.load(std::memory_order_acquire);
        auto cons = request_queue_control_->consumer_idx.load(std::memory_order_relaxed);

        if (prod != cons) {
            spdlog::trace("RequestReader: Processing queue state - producer_idx={}, consumer_idx={}, {} pending request(s)",
                        prod, cons, prod - cons);
        }

        while (cons != prod) {
            uint64_t slot_idx = cons % REQUEST_QUEUE_NUM_SLOTS;
            RequestSlot& slot = request_slots_[slot_idx];
            spdlog::trace("RequestReader: Processing slot {}, request_id={}, state={}",
                         slot_idx, slot.request_id, static_cast<int>(slot.state.load()));

            RequestState expected = RequestState::READY;
            if (!slot.state.compare_exchange_strong(expected,
                                                    RequestState::READING,
                                                    std::memory_order_acq_rel)) {
                spdlog::debug("RequestReader: Slot {} not in READY state, found state {} instead",
                             slot_idx, static_cast<int>(expected));
                break;
            }

            spdlog::debug("RequestReader: Processing request_id={} from slot {}", slot.request_id, slot_idx);

            /* ---- copy / build RawRequestData ---- */
            auto raw = std::make_unique<engine::RawRequestData>();
            raw->request_id           = slot.request_id;
            raw->arrival_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            raw->_shm_prompt_offset   = slot.prompt_shm_offset;
            raw->_shm_prompt_size     = slot.prompt_shm_size;

            // Read prompt from SHM into string
            try {
                spdlog::trace("RequestReader: Reading prompt from SHM offset={}, size={}",
                             slot.prompt_shm_offset, slot.prompt_shm_size);
                raw->prompt_payload = read_prompt_string(slot.prompt_shm_offset, slot.prompt_shm_size);
                spdlog::trace("RequestReader: Successfully read prompt of {} bytes", raw->prompt_payload.size());
            } catch (const std::exception& e) {
                spdlog::error("RequestReader: Failed to read prompt from SHM for request_id={}: {}",
                             slot.request_id, e.what());
                slot.state.store(RequestState::FREE, std::memory_order_release);
                break;
            }

            // Copy remaining parameters
            raw->sampling_params      = slot.sampling_params;
            raw->logits_params        = slot.logits_params;
            raw->stop_criteria        = slot.stop_criteria;
            raw->ipc_handles          = slot.ipc_handles;
            raw->tool_schemas_str     = slot.tool_schemas_str;
            raw->response_format_str  = slot.response_format_str;

            spdlog::debug("RequestReader: Request_id={} has temperature={}, top_p={}, max_tokens={}",
                         slot.request_id,
                         slot.sampling_params.temperature,
                         slot.sampling_params.top_p,
                         slot.stop_criteria.max_generated_tokens);

            // Push to preprocessor queue
            if (!output_queue_.push(std::move(raw))) {
                spdlog::error("RequestReader: RawRequestQueue full – dropping request_id={}", slot.request_id);
                slot.state.store(RequestState::FREE, std::memory_order_release);
                break;      // prevent producer from racing with us
            }

            spdlog::info("RequestReader: Successfully pushed request_id={} to preprocessor queue", slot.request_id);

            /* mark slot FREE */
            slot.state.store(RequestState::FREE, std::memory_order_release);
            ++cons;
            request_queue_control_->consumer_idx.store(cons, std::memory_order_release);
        }
    }

    std::string RequestReader::read_prompt_string(uint64_t offset, uint64_t size) {
        if (!bulk_data_map_ptr_) {
            spdlog::error("RequestReader: Null bulk_data_map_ptr_ in read_prompt_string");
            throw std::runtime_error("RequestReader: Null bulk data base address");
        }

        if (size == 0) {
            spdlog::warn("RequestReader: Zero-size prompt at offset {}", offset);
            return "";
        }

        char* base = static_cast<char*>(bulk_data_map_ptr_);
        spdlog::trace("RequestReader: Reading prompt string from bulk data offset {} with size {}", offset, size);
        return std::string(base + offset, size);
    }


}
