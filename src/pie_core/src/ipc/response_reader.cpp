#include "ipc/response_reader.hpp"
#include "ipc/response.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <stdexcept>

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/select.h>
#else
#warning "Kernel event notification for ResponseReader might not be implemented for this platform."
#endif

namespace pie_core::ipc {

    // --- Global Instance Management ---
    std::unique_ptr<ResponseReader> g_response_reader_instance = nullptr;

    ResponseReader* get_global_response_reader() {
        if (!g_response_reader_instance) {
            spdlog::warn("ResponseReader global instance accessed before initialization.");
            init_global_response_reader(); // lazy init
            if (!g_response_reader_instance) // Check again
                throw std::runtime_error("ResponseReader global instance not initialized.");
        }
        return g_response_reader_instance.get();
     }

     void init_global_response_reader(const std::string& name) {
        if (!g_response_reader_instance) {
            try {
                g_response_reader_instance = std::make_unique<ResponseReader>(name);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed to initialize global ResponseReader: ") + e.what());
            }
        }
     }

     void shutdown_global_response_reader() {
         g_response_reader_instance.reset();
     }
     // --- End Global Instance Management ---


    ResponseReader::ResponseReader(const std::string& response_shm_name)
        : response_shm_name_(response_shm_name)
    {
        if (!initialize_ipc_resources()) {
            throw std::runtime_error("ResponseReader: Failed to initialize IPC resources for " + response_shm_name_);
        }
        spdlog::info("ResponseReader: Initialized for SHM segment '{}'.", response_shm_name_);
         #if defined(__APPLE__)
            kernel_event_fd_ = -1; // Placeholder, polling used
            spdlog::debug("ResponseReader: Using polling mechanism (no kqueue init).");
         #elif defined(__linux__)
             kernel_event_fd_ = -1; // Placeholder
             spdlog::debug("ResponseReader: Using polling mechanism (no eventfd init).");
         #endif
    }

    ResponseReader::~ResponseReader() {
        cleanup_ipc_resources();
        if (kernel_event_fd_ != -1) {
            close(kernel_event_fd_);
        }
        spdlog::info("ResponseReader: Cleaned up resources for SHM segment '{}'.", response_shm_name_);
    }

    bool ResponseReader::initialize_ipc_resources() {
        response_shm_fd_ = shm_open(response_shm_name_.c_str(), O_RDWR, 0);
        if (response_shm_fd_ == -1) {
            spdlog::error("ResponseReader: shm_open failed for '{}': {}", response_shm_name_, strerror(errno));
            return false;
        }
        spdlog::debug("ResponseReader: Opened response SHM '{}', fd={}", response_shm_name_, response_shm_fd_);

        response_shm_map_ptr_ = mmap(nullptr, RESPONSE_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, response_shm_fd_, 0);
        if (response_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("ResponseReader: mmap failed for '{}': {}", response_shm_name_, strerror(errno));
            close(response_shm_fd_);
            response_shm_fd_ = -1;
            return false;
        }
        spdlog::debug("ResponseReader: Mapped response SHM '{}' at address {:p}", response_shm_name_, response_shm_map_ptr_);

        response_queue_control_ = static_cast<ResponseQueueControl*>(response_shm_map_ptr_);
        response_slots_ = reinterpret_cast<ResponseDeltaSlot*>(static_cast<char*>(response_shm_map_ptr_) + sizeof(ResponseQueueControl));

        return true;
    }

    void ResponseReader::cleanup_ipc_resources() {
         if (response_shm_map_ptr_ != nullptr && response_shm_map_ptr_ != MAP_FAILED) {
             if (munmap(response_shm_map_ptr_, RESPONSE_QUEUE_SHM_SIZE) == -1) {
                 spdlog::error("ResponseReader: munmap failed for '{}': {}", response_shm_name_, strerror(errno));
             }
             response_shm_map_ptr_ = nullptr;
         }
         if (response_shm_fd_ != -1) {
             if (close(response_shm_fd_) == -1) {
                 spdlog::error("ResponseReader: close failed for fd {}: {}", response_shm_fd_, strerror(errno));
             }
             response_shm_fd_ = -1;
         }
         response_queue_control_ = nullptr;
         response_slots_ = nullptr;
    }

    // --- Placeholder for kernel event waiting ---
    bool ResponseReader::wait_for_event(int timeout_ms) {
        if (timeout_ms > 0) {
             std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
        } else if (timeout_ms < 0) {
             std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    bool ResponseReader::consume_next_delta(ResponseDeltaSlot& out_delta, int timeout_ms) {
        if (!response_queue_control_ || !response_slots_) {
            throw std::runtime_error("ResponseReader: Response SHM not initialized.");
        }

        std::chrono::steady_clock::time_point deadline;
        bool use_timeout = (timeout_ms >= 0);
        if (use_timeout) {
            deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        }

        while (true) {
            uint64_t prod_idx = response_queue_control_->producer_idx.load(std::memory_order_acquire);
            uint64_t cons_idx = response_queue_control_->consumer_idx.load(std::memory_order_relaxed);

            if (cons_idx != prod_idx) {
                uint64_t current_slot_idx = cons_idx % RESPONSE_QUEUE_NUM_SLOTS;
                ResponseDeltaSlot& slot = response_slots_[current_slot_idx];

                ResponseSlotState expected_ready = ResponseSlotState::READY_FOR_PYTHON;
                if (slot.state.compare_exchange_strong(expected_ready, ResponseSlotState::PYTHON_READING,
                                                       std::memory_order_acq_rel, std::memory_order_relaxed))
                {

                    out_delta.request_id = slot.request_id;
                    out_delta.num_tokens_in_delta = slot.num_tokens_in_delta;
                    out_delta.is_final_delta = slot.is_final_delta;
                    out_delta.finish_reason = slot.finish_reason;

                    std::memcpy(
                        out_delta.tokens,
                        slot.tokens,
                        out_delta.num_tokens_in_delta * sizeof(int32_t)
                    );
                    std::memcpy(
                        out_delta.logprobs,
                        slot.logprobs,
                        out_delta.num_tokens_in_delta * MAX_LOGPROBS_PER_TOKEN * sizeof(float)
                    );

                    // Mark SHM slot as free for C++ producer after copying
                    slot.state.store(ResponseSlotState::FREE_FOR_CPP_WRITER, std::memory_order_release);
                    response_queue_control_->consumer_idx.store(cons_idx + 1, std::memory_order_release);

                    spdlog::trace("ResponseReader: Consumed delta for request {}", out_delta.request_id);
                    return true;
                }
                spdlog::trace("ResponseReader: Slot {} not ready or CAS failed (Expected READY, got {}). Retrying.", current_slot_idx, static_cast<int>(expected_ready));
            }

            if (use_timeout && std::chrono::steady_clock::now() >= deadline) {
                spdlog::trace("ResponseReader: Timeout reached.");
                return false;
            }

            int sleep_ms = 1; // Default sleep
            if(use_timeout && timeout_ms > 10) {
                sleep_ms = std::min(10, timeout_ms / 10); // Sleep up to 10ms, proportional to timeout
            } else if (use_timeout) {
                 sleep_ms = 1; // Minimum sleep for short timeouts
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
        return false; // Should never reach here
    }

} // namespace pie_core::ipc
