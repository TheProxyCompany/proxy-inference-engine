#include "ipc/response_writer.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

namespace pie_core::ipc {

    ResponseWriter::ResponseWriter(const std::string& response_shm_name)
        : response_shm_name_(response_shm_name)
    {
        if (!initialize_ipc_resources()) {
            throw ResponseWriterError("ResponseWriter: Failed to initialize IPC resources for " + response_shm_name_);
        }
        spdlog::info("ResponseWriter: Initialized for SHM segment '{}'.", response_shm_name_);
    }

    ResponseWriter::~ResponseWriter() {
        cleanup_ipc_resources();
        spdlog::info("ResponseWriter: Cleaned up resources for SHM segment '{}'.", response_shm_name_);
    }

    bool ResponseWriter::initialize_ipc_resources() {
        // 1. Open Existing SHM
        response_shm_fd_ = shm_open(response_shm_name_.c_str(), O_RDWR, 0);
        if (response_shm_fd_ == -1) {
            spdlog::error("ResponseWriter: shm_open failed for '{}': {}", response_shm_name_, strerror(errno));
            return false;
        }
        spdlog::debug("ResponseWriter: Opened response SHM '{}', fd={}", response_shm_name_, response_shm_fd_);

        // 2. Map SHM
        response_shm_map_ptr_ = mmap(nullptr, RESPONSE_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, response_shm_fd_, 0);
        if (response_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("ResponseWriter: mmap failed for '{}': {}", response_shm_name_, strerror(errno));
            close(response_shm_fd_);
            response_shm_fd_ = -1;
            return false;
        }
        spdlog::debug("ResponseWriter: Mapped response SHM '{}' at address {:p}", response_shm_name_, response_shm_map_ptr_);

        // 3. Assign Pointers
        response_queue_control_ = static_cast<ResponseQueueControl*>(response_shm_map_ptr_);
        response_slots_ = reinterpret_cast<ResponseDeltaSlot*>(static_cast<char*>(response_shm_map_ptr_) + sizeof(ResponseQueueControl));

        return true;
    }

    void ResponseWriter::cleanup_ipc_resources() {
        if (response_shm_map_ptr_ != nullptr && response_shm_map_ptr_ != MAP_FAILED) {
            if (munmap(response_shm_map_ptr_, RESPONSE_QUEUE_SHM_SIZE) == -1) {
                spdlog::error("ResponseWriter: munmap failed for '{}': {}", response_shm_name_, strerror(errno));
            }
            response_shm_map_ptr_ = nullptr;
        }
        if (response_shm_fd_ != -1) {
            if (close(response_shm_fd_) == -1) {
                spdlog::error("ResponseWriter: close failed for fd {}: {}", response_shm_fd_, strerror(errno));
            }
            response_shm_fd_ = -1;
        }
        response_queue_control_ = nullptr;
        response_slots_ = nullptr;
    }

    void ResponseWriter::write_delta(const ResponseDeltaSlot& delta) {
        if (!response_queue_control_ || !response_slots_) {
            throw ResponseWriterError("ResponseWriter: Response SHM not initialized.");
        }

        // 1. Claim a slot
        uint64_t producer_ticket = response_queue_control_->producer_idx.fetch_add(1, std::memory_order_acq_rel);
        uint64_t slot_idx = producer_ticket % RESPONSE_QUEUE_NUM_SLOTS;
        ResponseDeltaSlot& slot = response_slots_[slot_idx];

        // 2. Wait for the slot to become FREE_FOR_CPP_WRITER
        ResponseSlotState expected_free = ResponseSlotState::FREE_FOR_CPP_WRITER;
        int spin_count = 0;
        const int max_spins = 1000000; // ~1 second timeout, adjust as needed
        while (!slot.state.compare_exchange_weak(expected_free, ResponseSlotState::CPP_WRITING,
                                                 std::memory_order_acq_rel, std::memory_order_relaxed)) {
            expected_free = ResponseSlotState::FREE_FOR_CPP_WRITER; // Reset for next attempt
            spin_count++;
            if (spin_count > max_spins) {
                // Roll back producer index if we timed out
                response_queue_control_->producer_idx.fetch_sub(1, std::memory_order_relaxed);
                throw ResponseWriterError("ResponseWriter: Timeout waiting for a free response slot for request "
                                          + std::to_string(delta.request_id) + ". Python consumer might be stuck or queue full.");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        // 3. Write data to the slot
        // We can copy the entire struct as it's designed for this.
        // Ensure all relevant fields in the input 'delta' are correctly set by the caller (Scheduler).
        slot.request_id = delta.request_id;
        slot.num_tokens_in_delta = delta.num_tokens_in_delta;
        // Copy token and logprob arrays (adjust size based on num_tokens_in_delta)
        if (delta.num_tokens_in_delta > 0 && delta.num_tokens_in_delta <= MAX_TOKENS_PER_DELTA) {
            std::memcpy(slot.tokens, delta.tokens, delta.num_tokens_in_delta * sizeof(int32_t));
            // Assuming logprobs are always provided if tokens are; adjust logic if needed
             std::memcpy(slot.logprobs, delta.logprobs, delta.num_tokens_in_delta * MAX_LOGPROBS_PER_TOKEN * sizeof(float));
        } else if (delta.num_tokens_in_delta > MAX_TOKENS_PER_DELTA) {
             spdlog::warn("ResponseWriter: num_tokens_in_delta ({}) exceeds MAX_TOKENS_PER_DELTA ({}) for request {}. Truncating.",
                          delta.num_tokens_in_delta, MAX_TOKENS_PER_DELTA, delta.request_id);
             slot.num_tokens_in_delta = MAX_TOKENS_PER_DELTA; // Correct the count in the slot
             std::memcpy(slot.tokens, delta.tokens, MAX_TOKENS_PER_DELTA * sizeof(int32_t));
             std::memcpy(slot.logprobs, delta.logprobs, MAX_TOKENS_PER_DELTA * MAX_LOGPROBS_PER_TOKEN * sizeof(float));
        }
        slot.is_final_delta = delta.is_final_delta;
        slot.finish_reason = delta.finish_reason;

        // Ensure all writes are visible before updating the state
        std::atomic_thread_fence(std::memory_order_release);

        // 4. Mark the slot as READY_FOR_PYTHON
        slot.state.store(ResponseSlotState::READY_FOR_PYTHON, std::memory_order_release);

        // 5. Trigger kernel event (Optional - Python side currently polls)
        // If needed, add event triggering logic here, similar to RequestWriter,
        // potentially using a separate kernel event fd managed by IPCManager
        // for the response queue.

        spdlog::trace("ResponseWriter: Wrote delta for request {}", delta.request_id);
    }

} // namespace pie_core::ipc
