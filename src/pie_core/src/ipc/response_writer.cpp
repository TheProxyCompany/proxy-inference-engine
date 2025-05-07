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
        spdlog::info("ResponseWriter: Initializing for SHM segment '{}'", response_shm_name_);

        if (!initialize_ipc_resources()) {
            spdlog::critical("ResponseWriter: Failed to initialize IPC resources for '{}'", response_shm_name_);
            throw ResponseWriterError("ResponseWriter: Failed to initialize IPC resources for " + response_shm_name_);
        }

        spdlog::info("ResponseWriter: Successfully initialized for SHM segment '{}'", response_shm_name_);
    }

    ResponseWriter::~ResponseWriter() {
        spdlog::info("ResponseWriter: Destructor called for SHM segment '{}'", response_shm_name_);
        cleanup_ipc_resources();
        spdlog::debug("ResponseWriter: Destructor complete, resources cleaned up for SHM segment '{}'", response_shm_name_);
    }

    bool ResponseWriter::initialize_ipc_resources() {
        spdlog::debug("ResponseWriter: Initializing IPC resources for '{}'", response_shm_name_);

        // 1. Open Existing SHM
        response_shm_fd_ = shm_open(response_shm_name_.c_str(), O_RDWR, 0);
        if (response_shm_fd_ == -1) {
            spdlog::error("ResponseWriter: shm_open failed for '{}': {} (errno={})",
                         response_shm_name_, strerror(errno), errno);
            return false;
        }
        spdlog::debug("ResponseWriter: Opened response SHM '{}', fd={}", response_shm_name_, response_shm_fd_);

        // 2. Map SHM
        response_shm_map_ptr_ = mmap(nullptr, RESPONSE_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, response_shm_fd_, 0);
        if (response_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("ResponseWriter: mmap failed for '{}': {} (errno={})",
                         response_shm_name_, strerror(errno), errno);
            close(response_shm_fd_);
            response_shm_fd_ = -1;
            return false;
        }
        spdlog::debug("ResponseWriter: Mapped response SHM '{}' at address {:p}", response_shm_name_, response_shm_map_ptr_);

        // 3. Assign Pointers
        // Control block is at the beginning of the shared memory
        response_queue_control_ = static_cast<ResponseQueueControl*>(response_shm_map_ptr_);
        // Slots start after the control block
        response_slots_ = reinterpret_cast<ResponseDeltaSlot*>(
            static_cast<char*>(response_shm_map_ptr_) + sizeof(ResponseQueueControl)
        );
        spdlog::debug("ResponseWriter: Set up response_queue_control_ at {:p}, response_slots_ at {:p}",
                     (void*)response_queue_control_, (void*)response_slots_);

        // 4. Read initial queue state
        uint64_t initial_prod_idx = response_queue_control_->producer_idx.load(std::memory_order_acquire);
        uint64_t initial_cons_idx = response_queue_control_->consumer_idx.load(std::memory_order_acquire);
        spdlog::info("ResponseWriter: Initial response queue state - producer_idx={}, consumer_idx={}",
                    initial_prod_idx, initial_cons_idx);

        // 5. Verify slot states
        int free_slots = 0;
        for (size_t i = 0; i < RESPONSE_QUEUE_NUM_SLOTS; i++) {
            if (response_slots_[i].state.load(std::memory_order_relaxed) == ResponseSlotState::FREE_FOR_CPP_WRITER) {
                free_slots++;
            }
        }
        spdlog::info("ResponseWriter: Found {} free slots out of {} total slots",
                    free_slots, RESPONSE_QUEUE_NUM_SLOTS);

        return true;
    }

    void ResponseWriter::cleanup_ipc_resources() {
        spdlog::debug("ResponseWriter: Cleaning up IPC resources for '{}'", response_shm_name_);

        if (response_shm_map_ptr_ != nullptr && response_shm_map_ptr_ != MAP_FAILED) {
            spdlog::debug("ResponseWriter: Unmapping response SHM at {:p}", response_shm_map_ptr_);
            if (munmap(response_shm_map_ptr_, RESPONSE_QUEUE_SHM_SIZE) == -1) {
                spdlog::error("ResponseWriter: munmap failed for '{}': {} (errno={})",
                             response_shm_name_, strerror(errno), errno);
            }
            response_shm_map_ptr_ = nullptr;
        }

        if (response_shm_fd_ != -1) {
            spdlog::debug("ResponseWriter: Closing response SHM fd {}", response_shm_fd_);
            if (close(response_shm_fd_) == -1) {
                spdlog::error("ResponseWriter: close failed for fd {}: {} (errno={})",
                             response_shm_fd_, strerror(errno), errno);
            }
            response_shm_fd_ = -1;
        }

        response_queue_control_ = nullptr;
        response_slots_ = nullptr;

        spdlog::debug("ResponseWriter: IPC resources cleanup complete for '{}'", response_shm_name_);
    }

    void ResponseWriter::write_delta(const ResponseDeltaSlot& delta) {
        if (!response_queue_control_ || !response_slots_) {
            spdlog::critical("ResponseWriter: Attempted to write delta with uninitialized SHM resources");
            throw ResponseWriterError("ResponseWriter: Response SHM not initialized.");
        }

        spdlog::debug("ResponseWriter: Writing delta for request_id={}, num_tokens={}, is_final={}, finish_reason={}",
                     delta.request_id, delta.num_tokens_in_delta, delta.is_final_delta,
                     static_cast<int>(delta.finish_reason));

        // 1. Claim a slot
        uint64_t producer_ticket = response_queue_control_->producer_idx.fetch_add(1, std::memory_order_acq_rel);
        uint64_t slot_idx = producer_ticket % RESPONSE_QUEUE_NUM_SLOTS;
        ResponseDeltaSlot& slot = response_slots_[slot_idx];

        spdlog::trace("ResponseWriter: Claimed response slot {} (producer_ticket={})", slot_idx, producer_ticket);

        // 2. Wait for the slot to become FREE_FOR_CPP_WRITER
        ResponseSlotState expected_free = ResponseSlotState::FREE_FOR_CPP_WRITER;
        int spin_count = 0;
        const int max_spins = 1000000; // ~1 second timeout, adjust as needed

        // Log current slot state before attempting to acquire
        spdlog::trace("ResponseWriter: Current state of slot {} is {}",
                     slot_idx, static_cast<int>(slot.state.load(std::memory_order_relaxed)));

        while (!slot.state.compare_exchange_weak(expected_free, ResponseSlotState::CPP_WRITING,
                                                 std::memory_order_acq_rel, std::memory_order_relaxed)) {
            expected_free = ResponseSlotState::FREE_FOR_CPP_WRITER; // Reset for next attempt
            spin_count++;

            if (spin_count % 100000 == 0) {
                spdlog::warn("ResponseWriter: Still waiting for slot {} to become free for request_id={}, current state={}, spin_count={}",
                            slot_idx, delta.request_id, static_cast<int>(slot.state.load(std::memory_order_relaxed)), spin_count);
            }

            if (spin_count > max_spins) {
                // Roll back producer index if we timed out
                response_queue_control_->producer_idx.fetch_sub(1, std::memory_order_relaxed);
                spdlog::error("ResponseWriter: Timeout waiting for slot {} to become free for request_id={} after {} spins. "
                             "Current slot state is {}. Producer_idx rolled back to {}.",
                             slot_idx, delta.request_id, spin_count,
                             static_cast<int>(slot.state.load(std::memory_order_relaxed)),
                             response_queue_control_->producer_idx.load(std::memory_order_relaxed));
                throw ResponseWriterError("ResponseWriter: Timeout waiting for a free response slot for request "
                                          + std::to_string(delta.request_id) + ". Python consumer might be stuck or queue full.");
            }

            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        spdlog::trace("ResponseWriter: Successfully acquired slot {} for request_id={} after {} spins",
                     slot_idx, delta.request_id, spin_count);

        // 3. Write data to the slot
        spdlog::trace("ResponseWriter: Writing data to slot {} for request_id={}", slot_idx, delta.request_id);

        // Set basic fields
        slot.request_id = delta.request_id;
        slot.num_tokens_in_delta = delta.num_tokens_in_delta;

        // Copy token and logprob arrays (adjust size based on num_tokens_in_delta)
        if (delta.num_tokens_in_delta == 0) {
            spdlog::debug("ResponseWriter: Zero tokens in delta for request_id={}, is_final={}",
                         delta.request_id, delta.is_final_delta);
        } else if (delta.num_tokens_in_delta <= MAX_TOKENS_PER_DELTA) {
            spdlog::trace("ResponseWriter: Copying {} tokens and logprobs for request_id={}",
                         delta.num_tokens_in_delta, delta.request_id);

            std::memcpy(slot.tokens, delta.tokens, delta.num_tokens_in_delta * sizeof(int32_t));

            // Log first few tokens for debugging
            if (delta.num_tokens_in_delta > 0) {
                std::string token_preview;
                for (size_t i = 0; i < std::min(5UL, static_cast<size_t>(delta.num_tokens_in_delta)); i++) {
                    if (i > 0) token_preview += ", ";
                    token_preview += std::to_string(delta.tokens[i]);
                }
                spdlog::trace("ResponseWriter: First tokens in delta for request_id={}: [{}]",
                             delta.request_id, token_preview);
            }

            // Assuming logprobs are always provided if tokens are; adjust logic if needed
            std::memcpy(slot.logprobs, delta.logprobs, delta.num_tokens_in_delta * MAX_LOGPROBS_PER_TOKEN * sizeof(float));
        } else if (delta.num_tokens_in_delta > MAX_TOKENS_PER_DELTA) {
            spdlog::warn("ResponseWriter: num_tokens_in_delta ({}) exceeds MAX_TOKENS_PER_DELTA ({}) for request_id={}. Truncating.",
                         delta.num_tokens_in_delta, MAX_TOKENS_PER_DELTA, delta.request_id);

            slot.num_tokens_in_delta = MAX_TOKENS_PER_DELTA; // Correct the count in the slot
            std::memcpy(slot.tokens, delta.tokens, MAX_TOKENS_PER_DELTA * sizeof(int32_t));
            std::memcpy(slot.logprobs, delta.logprobs, MAX_TOKENS_PER_DELTA * MAX_LOGPROBS_PER_TOKEN * sizeof(float));
        }

        // Set finish fields
        slot.is_final_delta = delta.is_final_delta;
        slot.finish_reason = delta.finish_reason;

        if (delta.is_final_delta) {
            spdlog::info("ResponseWriter: Final delta for request_id={} with finish_reason={}",
                        delta.request_id, static_cast<int>(delta.finish_reason));
        }

        // Ensure all writes are visible before updating the state
        std::atomic_thread_fence(std::memory_order_release);

        // 4. Mark the slot as READY_FOR_PYTHON
        spdlog::trace("ResponseWriter: Marking slot {} as READY_FOR_PYTHON for request_id={}",
                     slot_idx, delta.request_id);
        slot.state.store(ResponseSlotState::READY_FOR_PYTHON, std::memory_order_release);

        // 5. Trigger kernel event (Optional - Python side currently polls)
        // If needed, add event triggering logic here, similar to RequestWriter,
        // potentially using a separate kernel event fd managed by IPCManager
        // for the response queue.

        spdlog::debug("ResponseWriter: Successfully wrote delta with {} tokens for request_id={} to slot {}, producer_idx now at {}",
                     delta.num_tokens_in_delta, delta.request_id, slot_idx,
                     response_queue_control_->producer_idx.load(std::memory_order_relaxed));
    }

} // namespace pie_core::ipc
