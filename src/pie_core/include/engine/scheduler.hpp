#pragma once

#include "engine/page_allocator.hpp"
#include "engine/request_preprocessor.hpp"
#include "ipc/response_writer.hpp"
#include "models/imodel.hpp"
#include "sequence/sequence.hpp"
#include "samplers/isampler.hpp"
#include "logit_processors/logit_processor.hpp"

#include <memory>
#include <cstdint>
#include <vector>
#include <list>
#include <unordered_map>
#include <random>
#include <thread>
#include <atomic>

namespace pie_core::engine {

    /**
     * @brief Orchestrates LLM inference requests, managing batching, KV cache, and stepping the model.
     */
    class Scheduler {
    public:
        /**
         * @brief Constructor. Initializes the scheduler with necessary components.
         * @param allocator Reference to the PageAllocator for KV cache management.
         * @param model Reference to the loaded model object.
         * @param processed_queue Reference to the queue providing new sequences.
         * @param response_writer Reference to the writer for sending results back via IPC.
         * @param max_num_seqs Max concurrent sequences the scheduler will manage.
         * @param max_tokens_in_batch Max total tokens per GPU batch.
         */
        Scheduler(
            PageAllocator& allocator,
            models::IModel& model,
            RequestPreprocessor::ProcessedSequenceQueue& processed_queue,
            ipc::ResponseWriter& response_writer,
            size_t max_num_seqs = 256,
            size_t max_tokens_in_batch = 4096
        );

        /**
         * @brief Destructor.
         */
        ~Scheduler();

        /**
         * @brief Runs the main scheduler loop. Will be called by Engine in its own thread.
         */
        void run_loop();

        /**
         * @brief Signals the scheduler to stop.
         */
        void stop();


        // --- Prevent Copying/Moving ---
        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;
        Scheduler(Scheduler&&) = delete;
        Scheduler& operator=(Scheduler&&) = delete;

    private:
        // --- Core Components (References) ---
        PageAllocator& allocator_;
        models::IModel& model_;
        RequestPreprocessor::ProcessedSequenceQueue& incoming_sequence_queue_;
        ipc::ResponseWriter& response_writer_;

        // --- Configuration ---
        const size_t max_num_seqs_;
        const size_t max_tokens_in_batch_;

        // --- Internal State ---
        std::list<std::unique_ptr<sequence::Sequence>> waiting_sequences_;
        std::unordered_map<uint64_t, std::unique_ptr<sequence::Sequence>> running_sequences_;
        // Add maps/lists for swapped, completed etc. as needed

        std::mt19937 rng_;

        std::atomic<bool> stop_flag_{false};


        // --- Private Methods ---

        /**
         * @brief Pulls new sequences from the incoming queue and adds them to the waiting list.
         */
        void ingest_new_sequences();

        /**
         * @brief Selects sequences for the next batch based on state and resources.
         * @return A pair containing vectors of sequence IDs for prefill and decode.
         */
        std::pair<std::vector<uint64_t>, std::vector<uint64_t>> select_batch();

        /**
         * @brief Attempts to allocate KV cache pages for a sequence.
         * @param seq The sequence requiring pages.
         * @return True if allocation succeeded, false otherwise.
         */
        bool allocate_pages_for_sequence(sequence::Sequence& seq);

        /**
         * @brief Builds the BatchDetails struct for the selected sequences.
         * @param prefill_seq_ids IDs of sequences in prefill state for this batch.
         * @param decode_seq_ids IDs of sequences in decode state for this batch.
         * @return The constructed BatchDetails object.
         */
        BatchDetails build_batch_details(
            const std::vector<uint64_t>& prefill_seq_ids,
            const std::vector<uint64_t>& decode_seq_ids
        );

        /**
         * @brief Processes the model output logits, samples next tokens, updates sequences.
         * @param logits Model output logits.
         * @param batch_details Details of the batch that produced the logits.
         */
        void process_batch_output(const mx::array& logits, const BatchDetails& batch_details);

         /**
         * @brief Frees KV cache pages associated with a completed or aborted sequence.
         * @param seq The sequence whose pages need to be freed.
         */
        void free_sequence_pages(const sequence::Sequence& seq);

         /**
         * @brief Handles completed, aborted, or swapped-out sequences.
         */
        void cleanup_finished_sequences();

    };

} // namespace pie_core::engine
