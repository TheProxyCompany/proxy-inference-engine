#include "engine/scheduler.hpp"
#include "engine/batch_details.hpp"
#include "sequence/sequence.hpp"
#include "samplers/sampler_factory.hpp"
#include "logit_processors/logit_processor_factory.hpp"
#include "ipc/response.hpp"
#include <spdlog/spdlog.h>
#include <chrono>

namespace pie_core::engine {

    Scheduler::Scheduler(
        PageAllocator& allocator,
        models::IModel& model,
        RequestPreprocessor::ProcessedSequenceQueue& processed_queue,
        ipc::ResponseWriter& response_writer,
        size_t max_num_seqs,
        size_t max_tokens_in_batch
    ) : allocator_(allocator),
        model_(model),
        incoming_sequence_queue_(processed_queue),
        response_writer_(response_writer),
        max_num_seqs_(max_num_seqs),
        max_tokens_in_batch_(max_tokens_in_batch),
        rng_(std::random_device{}()) // Seed the random number generator
    {
        spdlog::info("Scheduler: Initialized with max_num_seqs={}, max_tokens_in_batch={}", max_num_seqs_, max_tokens_in_batch_);
    }

    Scheduler::~Scheduler() {
        stop();
        spdlog::info("Scheduler: Destructed.");
    }

    void Scheduler::stop() {
        stop_flag_.store(true, std::memory_order_release);
        spdlog::debug("Scheduler: Stop signal received.");
    }

    void Scheduler::run_loop() {
        spdlog::info("Scheduler: Run loop entered.");
        while (!stop_flag_.load(std::memory_order_acquire)) {
            // 1. Ingest new sequences
            ingest_new_sequences();

            std::unique_ptr<sequence::Sequence> seq_to_process = nullptr;

            if (!running_sequences_.empty()) {
                auto it = running_sequences_.begin();
                seq_to_process = std::move(it->second);
                running_sequences_.erase(it);
            }

            if (seq_to_process) {
                spdlog::info("Scheduler: Mock processing sequence ID {}", seq_to_process->sequence_id);

                // Simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

                // Create mock response
                ipc::ResponseDeltaSlot delta;
                delta.request_id = seq_to_process->sequence_id;
                delta.num_tokens_in_delta = 1;
                delta.tokens[0] = 64000; // Example token ID
                // Add dummy logprobs if needed for testing Python side
                delta.logprobs[0][0] = -0.1f;
                delta.is_final_delta = true; // Send just one final delta
                delta.finish_reason = sequence::FinishReason::STOP;

                // Send mock response
                try {
                    response_writer_.write_delta(delta);
                    spdlog::info("Scheduler: Sent mock response for seq ID {}", seq_to_process->sequence_id);
                } catch (const ipc::ResponseWriterError& e) {
                    spdlog::error("Scheduler: Failed to write mock response delta for seq {}: {}", seq_to_process->sequence_id, e.what());
                }
            } else {
                // No sequences waiting/running, sleep briefly
                if (stop_flag_.load(std::memory_order_relaxed)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        spdlog::info("Scheduler: Run loop exited.");
    }

    void Scheduler::ingest_new_sequences() {
        std::unique_ptr<sequence::Sequence> seq_ptr;
        while (running_sequences_.size() < max_num_seqs_ && incoming_sequence_queue_.pop(seq_ptr)) {
            spdlog::debug("Scheduler: Ingested new sequence ID {}.", seq_ptr->sequence_id);
            seq_ptr->status = sequence::SequenceStatus::PREFILLING;
            running_sequences_.emplace(seq_ptr->sequence_id, std::move(seq_ptr));
        }
    }

    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> Scheduler::select_batch() {
        std::vector<uint64_t> prefill_seq_ids;
        std::vector<uint64_t> decode_seq_ids;

        return {prefill_seq_ids, decode_seq_ids};
    }

    bool Scheduler::allocate_pages_for_sequence(sequence::Sequence& seq) {
        // Calculate how many *new* pages are needed based on current length and page table size
        size_t current_len = seq.get_logical_len();
        size_t current_blocks = seq.page_table.size();
        size_t required_blocks = (current_len + TOKEN_CAPACITY_PER_PAGE) / TOKEN_CAPACITY_PER_PAGE; // +1 for next token

        if (required_blocks <= current_blocks) {
            return true; // Already has enough pages allocated
        }

        size_t num_pages_to_alloc = required_blocks - current_blocks;
        spdlog::trace("Scheduler: Seq {} needs {} new pages (current_len={}, current_blocks={}, required_blocks={})",
                      seq.sequence_id, num_pages_to_alloc, current_len, current_blocks, required_blocks);

        for (size_t i = 0; i < num_pages_to_alloc; ++i) {
            auto page_id_opt = allocator_.allocate_page();
            if (!page_id_opt) {
                spdlog::warn("Scheduler: Page allocation failed for seq {}. Allocator out of pages.", seq.sequence_id);
                // Free any pages allocated in this attempt (if any) - requires tracking
                // For simplicity now, just return false. A real implementation needs rollback.
                return false;
            }
            seq.append_page(page_id_opt.value());
        }
        spdlog::trace("Scheduler: Successfully allocated {} pages for seq {}", num_pages_to_alloc, seq.sequence_id);
        return true;
    }


    BatchDetails Scheduler::build_batch_details(
         const std::vector<uint64_t>& prefill_seq_ids,
         const std::vector<uint64_t>& decode_seq_ids
    ) {
        BatchDetails details;
        details.num_prefill_sequences = prefill_seq_ids.size();
        details.num_decode_sequences = decode_seq_ids.size();

        std::vector<mx::array> batch_token_ids;
        std::vector<mx::array> batch_positions;
        std::vector<mx::array> batch_block_tables; // Placeholder for consolidated table


        size_t total_tokens = 0;

        auto process_sequence = [&](uint64_t seq_id, bool is_prefill) {
            auto it = running_sequences_.find(seq_id);
            if (it == running_sequences_.end()) {
                 spdlog::error("Scheduler: Sequence ID {} not found in running_sequences_ during build_batch_details.", seq_id);
                 return; // Skip this sequence
            }
            sequence::Sequence& seq = *(it->second);

            int input_len = 0;
            mx::array tokens_to_process = mx::array({});
            mx::array positions_for_tokens = mx::array({});

            // if (is_prefill) {
            //      // Simple prefill: process the whole prompt
            //      tokens_to_process = mx::array(std::vector<int32_t>(seq.tokens.begin(), seq.tokens.begin() + seq.prompt_len));
            //      input_len = seq.prompt_len;
            //      positions_for_tokens = mx::arange(0, input_len);
            //      seq.status = sequence::SequenceStatus::DECODING; // Transition state after prefill
            //      spdlog::trace("Scheduler: Building prefill batch for seq {}, len={}", seq_id, input_len);
            // } else { // Decode
            //      tokens_to_process = mx::array({seq.tokens.back()}); // Process the last token
            //      input_len = 1;
            //      positions_for_tokens = mx::array({static_cast<int>(seq.get_logical_len() - 1)}); // Position is current length - 1
            //      spdlog::trace("Scheduler: Building decode batch for seq {}, pos={}", seq_id, positions_for_tokens.item<int>());
            // }

            batch_token_ids.push_back(tokens_to_process);
            batch_positions.push_back(positions_for_tokens);
            details.sequence_ids.push_back(seq_id);
            details.input_lengths.push_back(input_len);
            details.context_lengths.push_back(seq.get_logical_len() - input_len); // Length *before* this step
            total_tokens += input_len;

            // --- TODO: Build Consolidated Block Table ---
            // This is highly dependent on the kernel's expected format.
            // Example: Flatten page table for this sequence and add to a batch list
            // batch_block_tables.push_back(mx::array(seq.page_table));
        };

        for (uint64_t id : prefill_seq_ids) {
            process_sequence(id, true);
        }
        for (uint64_t id : decode_seq_ids) {
            process_sequence(id, false);
        }

        if (!batch_token_ids.empty()) {
            details.token_ids = mx::concatenate(batch_token_ids);
            details.positions = mx::concatenate(batch_positions);
            // details.consolidated_block_table = mx::concatenate(batch_block_tables); // Concatenate/stack as needed
            details.consolidated_block_table = mx::zeros({1}); // Placeholder!
        } else {
             // Handle empty batch case if necessary, though select_batch should prevent this
             details.token_ids = mx::array({});
             details.positions = mx::array({});
             details.consolidated_block_table = mx::array({});
        }

        details.total_tokens_in_step = total_tokens;

        spdlog::trace("Scheduler: Built batch details with {} total tokens.", total_tokens);
        return details;
    }

    void Scheduler::process_batch_output(const mx::array& logits, const BatchDetails& batch_details) {
        // Logits shape: [total_tokens_in_step, vocab_size]

        size_t current_token_offset = 0;
        for (size_t i = 0; i < batch_details.sequence_ids.size(); ++i) {
            uint64_t seq_id = batch_details.sequence_ids[i];
            int num_tokens_for_seq = batch_details.input_lengths[i];

            auto it = running_sequences_.find(seq_id);
             if (it == running_sequences_.end()) {
                 spdlog::error("Scheduler: Sequence ID {} from batch not found in running_sequences_ during output processing.", seq_id);
                 current_token_offset += num_tokens_for_seq;
                 continue;
             }
            sequence::Sequence& seq = *(it->second);

            // 1. Extract the relevant slice of logits for the *last* token of this sequence
            //    For prefill, we only care about the logit for the token *after* the prompt.
            //    For decode, there's only one token's logit.
            size_t logit_index = current_token_offset + num_tokens_for_seq - 1;
            mx::array seq_logits = mx::slice(
                logits,
                {(int)logit_index, 0},
                {(int)logit_index + 1, logits.shape(1)});

            // 2. Apply Logit Processors
            std::vector<std::unique_ptr<logit_processors::ILogitProcessor>> processors =
                logit_processors::create_processors(seq.logits_params);
            for(const auto& processor : processors) {
                seq_logits = processor->process_logits(seq_logits, seq.logits_params, seq);
            }

            // 3. Sample Next Token
            std::unique_ptr<samplers::ISampler> sampler = samplers::create_sampler(seq.sampling_params);
            mx::array next_token_id_array = sampler->next_token(seq_logits, seq.sampling_params, rng_);
            int32_t next_token_id = next_token_id_array.item<int32_t>(); // Assuming sampler returns scalar array

            // 4. Update Sequence State
            seq.append_token(next_token_id);
            spdlog::trace("Scheduler: Appended token {} to seq {}", next_token_id, seq_id);

            // 5. Check Stop Conditions
            bool finished = false;
            sequence::FinishReason reason = sequence::FinishReason::STOP; // Default assumption
            if (seq.get_generation_len() >= seq.stop_criteria.max_generated_tokens) {
                 finished = true;
                 reason = sequence::FinishReason::LENGTH;
                 spdlog::debug("Scheduler: Seq {} finished due to length.", seq_id);
            } else {
                 // Check stop tokens
                 for (int32_t stop_id : seq.stop_criteria.stop_token_ids) {
                      if (next_token_id == stop_id) {
                           finished = true;
                           reason = sequence::FinishReason::STOP;
                           spdlog::debug("Scheduler: Seq {} finished due to stop token {}.", seq_id, stop_id);
                           break;
                      }
                 }
                 // TODO: Check for other stop reasons (user sequence, tool use signal)
            }

            // 6. Send Delta via IPC
            ipc::ResponseDeltaSlot delta;
            delta.request_id = seq.sequence_id; // Use sequence_id as request_id correlation
            delta.num_tokens_in_delta = 1;
            delta.tokens[0] = next_token_id;
            // TODO: Populate logprobs if requested
            delta.is_final_delta = finished;
            delta.finish_reason = reason;

            try {
                 response_writer_.write_delta(delta);
            } catch (const ipc::ResponseWriterError& e) {
                 spdlog::error("Scheduler: Failed to write response delta for seq {}: {}", seq_id, e.what());
                 // Mark sequence as errored?
                 seq.status = sequence::SequenceStatus::ERROR;
                 finished = true; // Mark as finished to trigger cleanup
            }

            // 7. Update sequence status if finished
            if (finished) {
                 seq.status = sequence::SequenceStatus::COMPLETED; // Or ERROR if write failed
            }

            // Move offset for the next sequence in the batch
            current_token_offset += num_tokens_for_seq;
        }
    }

    void Scheduler::free_sequence_pages(const sequence::Sequence& seq) {
        spdlog::debug("Scheduler: Freeing {} pages for sequence {}", seq.page_table.size(), seq.sequence_id);
        for (uint32_t page_id : seq.page_table) {
            allocator_.free_page(page_id);
        }
    }

    void Scheduler::cleanup_finished_sequences() {
        auto it = running_sequences_.begin();
        while (it != running_sequences_.end()) {
             sequence::Sequence& seq = *(it->second);
             if (seq.status == sequence::SequenceStatus::COMPLETED ||
                 seq.status == sequence::SequenceStatus::ERROR ||
                 seq.cancelled.load(std::memory_order_acquire))
             {
                  spdlog::info("Scheduler: Cleaning up finished/cancelled sequence ID {}. Status: {}", seq.sequence_id, static_cast<int>(seq.status));
                  free_sequence_pages(seq);
                  it = running_sequences_.erase(it); // Remove from running map
             } else {
                  ++it;
             }
        }
    }

} // namespace pie_core::engine
