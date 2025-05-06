#include "ipc/ipc_reader.hpp"
#include "ipc/shared_memory_manager.hpp" // For deallocating from bulk SHM

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread> // For std::this_thread::sleep_for

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/eventfd.h>
#endif

namespace pie_core::ipc {

    // Placeholder for SPSCQueue - replace with actual implementation
    template<typename T>
    class SPSCQueue {
    public:
        bool try_enqueue(T item) { return false; } // Placeholder
        bool try_dequeue(T& item) { return false; } // Placeholder
    };


    IPCReader::IPCReader(
        SequenceQueueType& output_queue,
        const std::string& request_shm_name,
        int kernel_event_fd)
        : output_queue_(output_queue),
          kernel_event_fd_(kernel_event_fd)
          // request_shm_name_ and bulk_shm_name_ are implicitly using defaults from ipc_request.hpp / ipc_reader.hpp
    {
        // Store names if they are passed, otherwise use defaults
        // This part is a bit redundant if we always use the const char* from headers,
        // but good if IPCReader could be configured with different names.
        // For now, let's assume we'll use the const char* names directly in initialize.
        if (!initialize_ipc_resources()) {
            throw std::runtime_error("IPCReader: Failed to initialize IPC resources.");
        }
        spdlog::info("IPCReader constructed and IPC resources initialized.");
    }

    IPCReader::~IPCReader() {
        stop(); // Ensure the run loop is stopped if not already
        cleanup_ipc_resources();
        spdlog::info("IPCReader destroyed.");
    }

    bool IPCReader::initialize_ipc_resources() {
        spdlog::info("IPCReader: Initializing IPC resources...");

        // 1. Request Queue SHM (Open Existing, created by main engine process)
        request_shm_fd_ = shm_open(REQUEST_QUEUE_SHM_NAME, O_RDWR, 0);
        if (request_shm_fd_ == -1) {
            spdlog::error("IPCReader: shm_open for request queue '{}' failed: {}", REQUEST_QUEUE_SHM_NAME, strerror(errno));
            return false;
        }
        request_shm_map_ptr_ = mmap(nullptr, REQUEST_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, request_shm_fd_, 0);
        if (request_shm_map_ptr_ == MAP_FAILED) {
            spdlog::error("IPCReader: mmap for request queue '{}' failed: {}", REQUEST_QUEUE_SHM_NAME, strerror(errno));
            close(request_shm_fd_); request_shm_fd_ = -1;
            return false;
        }
        request_queue_control_ = static_cast<RequestQueueControl*>(request_shm_map_ptr_);
        request_slots_ = reinterpret_cast<RequestSlot*>(static_cast<char*>(request_shm_map_ptr_) + sizeof(RequestQueueControl));
        spdlog::info("IPCReader: Request queue SHM '{}' opened and mapped.", REQUEST_QUEUE_SHM_NAME);

        // 2. Bulk Data SHM (Open Existing, created by main engine process)
        // The IPCReader needs to deallocate from this, so it needs a SharedMemoryManager
        // This assumes the main engine process creates one with create_if_not_exists=true
        // and this IPCReader will use one with create_if_not_exists=false.
        // However, the deallocation should happen via the *engine's* SMM instance.
        // For now, let's just map it for reading. Deallocation responsibility needs clarification.
        // Option 1: IPCReader has its own SMM to open it.
        // Option 2: IPCReader gets a raw mapped pointer from the engine, and calls a deallocate method
        //           on the engine's SMM instance. (Option 2 is cleaner for ownership of deallocation).

        // Let's assume for now IPCReader just maps it for reading, and deallocation is handled
        // via a reference to the engine's main SMM for bulk data.
        // This IPCReader instance will not *own* a SharedMemoryManager for the bulk data.
        // It will just need the mapped pointer.
        bulk_data_shm_fd_ = shm_open(BULK_DATA_SHM_NAME, O_RDWR, 0); // Read-write for now, could be read-only if only reading
        if (bulk_data_shm_fd_ == -1) {
            spdlog::error("IPCReader: shm_open for bulk data SHM '{}' failed: {}", BULK_DATA_SHM_NAME, strerror(errno));
            cleanup_ipc_resources(); // Clean up already opened request_shm
            return false;
        }
        bulk_data_map_ptr_ = mmap(nullptr, BULK_DATA_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bulk_data_shm_fd_, 0);
        if (bulk_data_map_ptr_ == MAP_FAILED) {
            spdlog::error("IPCReader: mmap for bulk data SHM '{}' failed: {}", BULK_DATA_SHM_NAME, strerror(errno));
            cleanup_ipc_resources(); // Clean up already opened request_shm and bulk_data_shm_fd
            return false;
        }
        spdlog::info("IPCReader: Bulk data SHM '{}' opened and mapped.", BULK_DATA_SHM_NAME);

        // Ensure kernel_event_fd_ is valid (passed from main engine setup)
        if (kernel_event_fd_ == -1) {
            spdlog::error("IPCReader: Invalid kernel_event_fd provided.");
            cleanup_ipc_resources();
            return false;
        }

        return true;
    }

    void IPCReader::cleanup_ipc_resources() {
        spdlog::info("IPCReader: Cleaning up IPC resources...");
        if (bulk_data_map_ptr_ != nullptr && bulk_data_map_ptr_ != MAP_FAILED) {
            munmap(bulk_data_map_ptr_, BULK_DATA_SHM_SIZE);
            bulk_data_map_ptr_ = nullptr;
        }
        if (bulk_data_shm_fd_ != -1) {
            close(bulk_data_shm_fd_);
            bulk_data_shm_fd_ = -1;
        }

        if (request_shm_map_ptr_ != nullptr && request_shm_map_ptr_ != MAP_FAILED) {
            munmap(request_shm_map_ptr_, REQUEST_QUEUE_SHM_SIZE);
            request_shm_map_ptr_ = nullptr;
        }
        if (request_shm_fd_ != -1) {
            close(request_shm_fd_);
            request_shm_fd_ = -1;
        }
        request_queue_control_ = nullptr;
        request_slots_ = nullptr;
    }

    void IPCReader::stop() {
        running_.store(false, std::memory_order_release);
        // TODO: May need to send a dummy event to wake up wait_for_notification if it's blocking indefinitely.
        // Or ensure wait_for_notification has a timeout.
    }

    bool IPCReader::wait_for_notification() {
        // This is a blocking call.
        // Add a timeout to allow the 'running_' flag to be checked periodically.
        #if defined(__APPLE__)
            struct kevent event;
            struct timespec timeout_ts = {1, 0}; // 1 second timeout
            int ret = kevent(kernel_event_fd_, nullptr, 0, &event, 1, &timeout_ts);
            if (ret == -1) {
                if (errno == EINTR) return true; // Interrupted by signal, check running_
                spdlog::error("IPCReader: kevent wait failed: {}", strerror(errno));
                return false; // Error
            }
            return ret > 0; // Event occurred if ret > 0
        #elif defined(__linux__)
            // eventfd read is blocking. Use poll/epoll with a timeout for non-blocking check of running_
            // For simplicity, a direct read with a flag check after might be okay if stop() is rare.
            // A more robust way: use epoll_wait with a timeout.
            // This simple version will block until an event or error.
            uint64_t val;
            ssize_t s = read(kernel_event_fd_, &val, sizeof(uint64_t));
            if (s == -1) {
                if (errno == EINTR || errno == EAGAIN) return true; // Interrupted or would block (if non-blocking), check running_
                spdlog::error("IPCReader: eventfd read failed: {}", strerror(errno));
                return false; // Error
            }
            return s == sizeof(uint64_t); // Event occurred
        #endif
        return false; // Should not reach here
    }

    void IPCReader::run() {
        running_.store(true, std::memory_order_relaxed);
        spdlog::info("IPCReader: Run loop started.");

        // TODO: Instantiate C++ Tokenizer here if IPCReader is responsible for it.
        // tokenizers::Tokenizer cxx_tokenizer("path/to/tokenizer.json"); // Example

        while (running_.load(std::memory_order_acquire)) {
            if (wait_for_notification()) { // Blocks until event or timeout
                process_incoming_requests();
            }
            // Loop continues if wait_for_notification timed out or was interrupted,
            // allowing running_ flag to be checked.
        }
        spdlog::info("IPCReader: Run loop finished.");
    }

    void IPCReader::process_incoming_requests() {
        if (!request_queue_control_ || !request_slots_) {
            spdlog::error("IPCReader: Request queue SHM not initialized in process_incoming_requests.");
            return;
        }

        uint64_t current_consumer_idx = request_queue_control_->consumer_idx.load(std::memory_order_relaxed);
        uint64_t producer_idx_snapshot = request_queue_control_->producer_idx.load(std::memory_order_acquire);

        while (current_consumer_idx < producer_idx_snapshot) {
            if (!running_.load(std::memory_order_acquire)) break; // Check running flag frequently

            uint64_t slot_idx = current_consumer_idx % REQUEST_QUEUE_NUM_SLOTS;
            RequestSlot& slot = request_slots_[slot_idx];
            RequestState expected_ready = RequestState::READY;

            // Attempt to claim the slot
            if (slot.state.compare_exchange_strong(expected_ready, RequestState::READING,
                                                   std::memory_order_acq_rel, std::memory_order_relaxed)) {
                // Successfully claimed slot
                spdlog::debug("IPCReader: Processing request ID {} from slot {}", slot.request_id, slot_idx);

                try {
                    std::unique_ptr<sequence::Sequence> seq = build_sequence_from_slot(slot);
                    if (seq) {
                        // Enqueue to Scheduler's SPSC queue
                        if (!output_queue_.try_enqueue(std::move(seq))) {
                            spdlog::error("IPCReader: Failed to enqueue sequence ID {} to scheduler queue (full?). Reverting slot state.", slot.request_id);
                            // Revert state, tricky. Better if SPSC queue can block or has sufficient capacity.
                            // For now, we "lose" this request from IPC perspective if SPSC is full.
                            // This needs robust handling - e.g. retry, or SPSC queue that blocks/signals.
                            slot.state.store(RequestState::READY, std::memory_order_release); // Put it back for now
                            break; // Stop processing this batch to avoid further SPSC queue issues
                        } else {
                            spdlog::debug("IPCReader: Enqueued sequence ID {} to scheduler.", slot.request_id);
                        }
                    } else {
                        spdlog::error("IPCReader: Failed to build sequence from slot {} for request ID {}", slot_idx, slot.request_id);
                        // Error already logged by build_sequence_from_slot
                    }
                } catch (const std::exception& e) {
                    spdlog::error("IPCReader: Exception building sequence for request ID {}: {}", slot.request_id, e.what());
                    // Sequence object might not be created, slot needs to be freed.
                }

                // Mark slot as FREE for Python producer, regardless of sequence build success/failure
                // to prevent producer from stalling. Errors should be handled or logged.
                slot.state.store(RequestState::FREE, std::memory_order_release);
                request_queue_control_->consumer_idx.fetch_add(1, std::memory_order_release); // Advance consumer index
                current_consumer_idx++;

            } else if (expected_ready == RequestState::FREE || expected_ready == RequestState::WRITING) {
                // Slot was not ready, or producer is still writing. This is fine, producer hasn't finished.
                // We caught up to the producer or a slot that's being written.
                break;
            } else if (expected_ready == RequestState::READING) {
                // This should not happen if IPCReader is single-threaded.
                // Indicates a bug or another consumer.
                spdlog::error("IPCReader: Slot {} is already in READING state! Possible race condition or bug.", slot_idx);
                current_consumer_idx++; // Skip this problematic slot
            } else {
                // Unknown state
                 spdlog::warn("IPCReader: Slot {} has unexpected state {}", slot_idx, static_cast<uint32_t>(expected_ready));
                 current_consumer_idx++; // Skip
            }
        }
    }

    std::vector<int32_t> IPCReader::read_prompt_tokens(uint64_t offset, uint64_t size) {
        // THIS FUNCTION IS NOW FOR RAW STRING DATA, NOT TOKEN IDS
        if (!bulk_data_map_ptr_) {
            spdlog::error("IPCReader: Bulk data SHM not mapped for reading prompt string.");
            return {}; // Return empty or throw
        }
        if (offset + size > BULK_DATA_SHM_SIZE) { // Basic bounds check
            spdlog::error("IPCReader: Prompt SHM read out of bounds (offset={}, size={}, limit={})", offset, size, BULK_DATA_SHM_SIZE);
            return {};
        }

        // Create a string_view or copy into a std::string
        // For tokenization, a std::string is usually easier to pass to tokenizer libraries.
        const char* prompt_char_ptr = static_cast<const char*>(bulk_data_map_ptr_) + offset;
        std::string prompt_string(prompt_char_ptr, size);

        // TODO: Tokenization will happen here or in build_sequence_from_slot
        // For now, this function's purpose shifts to returning the raw string for tokenization.
        // So, the return type should ideally be std::string.
        // To keep the signature for now and illustrate the point:
        spdlog::debug("IPCReader: Read raw prompt string of size {} from SHM offset {}", size, offset);

        // Placeholder: This function should now return std::string.
        // The tokenization will happen in build_sequence_from_slot.
        // For now, returning empty vector as tokenization isn't done here.
        // This function should be renamed to read_prompt_string.
        // For the purpose of this draft, let's simulate it returning an empty token list.
        // In reality, you'd return the string.
        return {}; // This is incorrect now, should return string
    }


    std::unique_ptr<sequence::Sequence> IPCReader::build_sequence_from_slot(const RequestSlot& slot) {
        spdlog::debug("IPCReader: Building sequence for request ID {}", slot.request_id);
        if (!bulk_data_map_ptr_) {
            spdlog::error("IPCReader: Bulk data SHM not mapped. Cannot build sequence for request ID {}.", slot.request_id);
            return nullptr;
        }
        if (slot.prompt_shm_offset + slot.prompt_shm_size > BULK_DATA_SHM_SIZE) {
            spdlog::error("IPCReader: Invalid SHM offset/size for prompt for request ID {}.", slot.request_id);
            return nullptr;
        }

        const char* prompt_char_ptr = static_cast<const char*>(bulk_data_map_ptr_) + slot.prompt_shm_offset;
        std::string prompt_string(prompt_char_ptr, slot.prompt_shm_size);

        // --- TOKENIZATION HAPPENS HERE ---
        std::vector<int32_t> token_ids;
        // Example:
        // if (cxx_tokenizer_) { // Assuming cxx_tokenizer_ is a member or accessible
        //    token_ids = cxx_tokenizer_->encode(prompt_string);
        // } else {
        //    spdlog::error("IPCReader: C++ tokenizer not available for request ID {}.", slot.request_id);
        //    return nullptr;
        // }
        // Placeholder for tokenization:
        if (prompt_string == "hello") token_ids = {1, 2, 3};
        else if (prompt_string == "world") token_ids = {4, 5, 6};
        else token_ids = {7, 8, 9}; // Dummy tokenization
        // --- END TOKENIZATION ---

        if (token_ids.empty() && !prompt_string.empty()) {
             spdlog::warn("IPCReader: Tokenization resulted in empty token list for non-empty prompt for request ID {}.", slot.request_id);
             // Decide if this is an error or acceptable
        }

        // Deallocate the prompt string from bulk SHM
        // This requires access to the engine's SharedMemoryManager instance
        // that created/manages the bulk_data_shm.
        // For now, this is a conceptual call.
        // engine_smm_instance->deallocate(static_cast<void*>(prompt_char_ptr_from_base));
        spdlog::debug("IPCReader: TODO - Deallocate prompt string from SHM for request ID {} (offset {}, size {}).",
                      slot.request_id, slot.prompt_shm_offset, slot.prompt_shm_size);


        // Create Sequence object
        // TODO: Get arrival_timestamp_ns properly
        uint64_t arrival_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        try {
            return std::make_unique<sequence::Sequence>(
                slot.request_id,
                sequence::SequenceStatus::WAITING,
                arrival_timestamp,
                token_ids,
                token_ids.size(), // prompt_len is now full tokenized length
                slot.sampling_params,
                slot.logits_params,
                slot.stop_criteria,
                slot.ipc_handles
            );
        } catch (const std::exception& e) {
            spdlog::error("IPCReader: Failed to create Sequence object for request ID {}: {}", slot.request_id, e.what());
            return nullptr;
        }
    }

} // namespace pie_core::ipc
