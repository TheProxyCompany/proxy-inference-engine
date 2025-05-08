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
        PostprocessingQueue& postprocessing_queue,
        ipc::ResponseWriter& response_writer,
        size_t max_num_seqs,
        size_t max_tokens_in_batch
    ) : allocator_(allocator),
        model_(model),
        incoming_sequence_queue_(processed_queue),
        postprocessing_queue_(postprocessing_queue),
        response_writer_(response_writer),
        max_num_seqs_(max_num_seqs),
        max_tokens_in_batch_(max_tokens_in_batch),
        rng_(std::random_device{}()) // Seed the random number generator
    {
        spdlog::info("Scheduler: Initializing with max_num_seqs={}, max_tokens_in_batch={}",
                     max_num_seqs_, max_tokens_in_batch_);

        // Log model information
        spdlog::info("Scheduler: Using model with {} layers, {} KV heads, {} hidden dim, {} vocab size",
                    model_.get_num_layers(), model_.get_num_kv_heads(),
                    model_.get_head_dim(), model_.get_vocab_size());

        // Log allocator information
        spdlog::info("Scheduler: Using PageAllocator with {} free pages available",
                    allocator_.get_num_free_pages());

        spdlog::info("Scheduler: Initialization complete");
    }

    Scheduler::~Scheduler() {
        spdlog::info("Scheduler: Destructor called");
        stop();

        // Log status of running sequences during destruction
        if (!running_sequences_.empty()) {
            spdlog::warn("Scheduler: Destructor called with {} active sequences still running",
                        running_sequences_.size());

            // Log information about each running sequence
            for (const auto& [seq_id, seq_ptr] : running_sequences_) {
                spdlog::debug("Scheduler: During destruction, abandoning sequence_id={} with status={}, logical_len={}",
                            seq_id, static_cast<int>(seq_ptr->status), seq_ptr->get_logical_len());
            }
        }

        spdlog::info("Scheduler: Destructor complete");
    }

    void Scheduler::stop() {
        bool was_already_stopping = stop_flag_.exchange(true, std::memory_order_acq_rel);
        if (!was_already_stopping) {
            spdlog::info("Scheduler: Stop signal received");
        } else {
            spdlog::debug("Scheduler: Duplicate stop signal received (already stopping)");
        }
    }

    void Scheduler::set_attention_type(AttentionType type) {
        attention_type_ = type;
        spdlog::info("Scheduler: Attention type set to {}",
                   type == AttentionType::STANDARD ? "STANDARD" : "PAGED");
    }

    void Scheduler::run_loop() {
        spdlog::info("Scheduler: Run loop entered");

        // Track statistics
        uint64_t loop_counter = 0;
        uint64_t sequences_ingested = 0;
        uint64_t sequences_processed = 0;
        uint64_t tokens_generated = 0;

        auto start_time = std::chrono::steady_clock::now();

        while (!stop_flag_.load(std::memory_order_acquire)) {
            loop_counter++;

            // Log occasional statistics at trace level
            if (loop_counter % 1000 == 0) {
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();

                double tokens_per_second = (elapsed_ms > 0) ?
                    static_cast<double>(tokens_generated) / (static_cast<double>(elapsed_ms) / 1000.0) : 0.0;

                spdlog::trace("Scheduler: Stats - iteration {}, {} seqs ingested, {} seqs processed, {} tokens @ {:.2f} tokens/sec",
                             loop_counter, sequences_ingested, sequences_processed, tokens_generated, tokens_per_second);

                spdlog::trace("Scheduler: Currently have {} sequences running, {} free pages in allocator",
                             running_sequences_.size(), allocator_.get_num_free_pages());
            }

            // 1. Ingest new sequences
            ingest_new_sequences();


            // --- Mock sequence processing for now ---
            std::unique_ptr<sequence::Sequence> seq_to_process = nullptr;

            if (!running_sequences_.empty()) {
                auto it = running_sequences_.begin();
                seq_to_process = std::move(it->second);
                running_sequences_.erase(it);
                spdlog::debug("Scheduler: Selected sequence_id={} for processing, {} sequences remaining",
                             seq_to_process->sequence_id, running_sequences_.size());
            }

            if (seq_to_process) {
                sequences_processed++;
                uint64_t seq_id = seq_to_process->sequence_id;

                spdlog::info("Scheduler: Processing sequence_id={} (sequence #{} in this session)",
                           seq_id, sequences_processed);

                // Log detailed sequence information at debug level
                spdlog::debug("Scheduler: Sequence_id={} details: status={}, prompt_len={}, tokens_generated={}, temperature={}, max_tokens={}",
                            seq_id,
                            static_cast<int>(seq_to_process->status),
                            seq_to_process->prompt_len,
                            seq_to_process->get_generation_len(),
                            seq_to_process->sampling_params.temperature,
                            seq_to_process->stop_criteria.max_generated_tokens);

                // Simulate work
                auto start_processing = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

                // Track token generation
                tokens_generated += 1; // We're generating one token in this mock implementation

                // Create mock postprocessing data
                std::unique_ptr<PostprocessingData> pp_data = std::make_unique<PostprocessingData>();
                pp_data->request_id = seq_id;
                pp_data->next_token_id = 64000; // Example token ID
                pp_data->is_final_delta = true; // Send just one final delta
                pp_data->finish_reason = sequence::FinishReason::STOP;

                // Send to postprocessing queue
                try {
                    spdlog::debug("Scheduler: Sending token to postprocessor for sequence_id={}", seq_id);
                    if (postprocessing_queue_.push(std::move(pp_data))) {
                        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start_processing).count();

                        spdlog::info("Scheduler: Successfully queued token for postprocessing for sequence_id={} in {}ms",
                                   seq_id, processing_time);
                    } else {
                        spdlog::error("Scheduler: Failed to queue token for postprocessing for sequence_id={} (queue full)",
                                    seq_id);

                        // Fallback to direct response writing
                        ipc::ResponseDeltaSlot delta;
                        delta.request_id = seq_id;
                        delta.num_tokens_in_delta = 1;
                        delta.tokens[0] = 64000; // Example token ID
                        delta.logprobs[0][0] = -0.1f;
                        delta.is_final_delta = true;
                        delta.finish_reason = sequence::FinishReason::STOP;
                        std::string fallback_content = "Hello, world!";
                        std::memcpy(delta.content, fallback_content.c_str(), fallback_content.size());
                        delta.content[fallback_content.size()] = '\0';
                        delta.content_len = fallback_content.size();

                        spdlog::warn("Scheduler: Fallback - Direct response writing for sequence_id={}", seq_id);
                        response_writer_.write_delta(delta);
                    }
                } catch (const std::exception& e) {
                    spdlog::error("Scheduler: Error processing sequence_id={}: {}", seq_id, e.what());
                }
            } else {
                // No sequences waiting/running, sleep briefly
                if (stop_flag_.load(std::memory_order_relaxed)) {
                    spdlog::debug("Scheduler: Stop flag detected, exiting run loop");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Calculate final statistics for run loop
        auto total_runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        double tokens_per_second = (total_runtime_ms > 0) ?
            static_cast<double>(tokens_generated) / (static_cast<double>(total_runtime_ms) / 1000.0) : 0.0;

        spdlog::info("Scheduler: Run loop exited after {} iterations in {}ms", loop_counter, total_runtime_ms);
        spdlog::info("Scheduler: Processed {} sequences, generated {} tokens @ {:.2f} tokens/sec",
                   sequences_processed, tokens_generated, tokens_per_second);
    }

    void Scheduler::ingest_new_sequences() {
        size_t ingested_count = 0;
        size_t start_size = running_sequences_.size();
        size_t capacity = max_num_seqs_ - start_size;

        if (capacity == 0) {
            // Already at max capacity
            spdlog::trace("Scheduler: At maximum capacity ({} sequences), no new sequences will be ingested", max_num_seqs_);
            return;
        }

        spdlog::trace("Scheduler: Attempting to ingest new sequences (capacity: {}/{})",
                     start_size, max_num_seqs_);

        std::unique_ptr<sequence::Sequence> seq_ptr;
        while (running_sequences_.size() < max_num_seqs_ && incoming_sequence_queue_.pop(seq_ptr)) {
            uint64_t seq_id = seq_ptr->sequence_id;

            spdlog::debug("Scheduler: Ingesting new sequence_id={}, state: {}, prompt_len: {}",
                         seq_id, static_cast<int>(seq_ptr->status), seq_ptr->prompt_len);

            // Transition to PREFILLING state
            sequence::SequenceStatus old_status = seq_ptr->status;
            seq_ptr->status = sequence::SequenceStatus::PREFILLING;

            spdlog::trace("Scheduler: Transitioned sequence_id={} from status {} to {}",
                         seq_id, static_cast<int>(old_status), static_cast<int>(seq_ptr->status));

            // Check if we need to allocate pages
            size_t page_count_before = seq_ptr->page_table.size();
            size_t tokens_to_process = seq_ptr->prompt_len;

            size_t required_pages = (tokens_to_process + TOKEN_CAPACITY_PER_PAGE - 1) / TOKEN_CAPACITY_PER_PAGE;
            if (required_pages > page_count_before) {
                spdlog::debug("Scheduler: Sequence_id={} needs {} pages for {} tokens (has {} pages currently)",
                             seq_id, required_pages, tokens_to_process, page_count_before);

                // In a real implementation, we would allocate pages here using allocate_pages_for_sequence
                // For now, just log it
            }

            // Store the sequence in our active map
            running_sequences_.emplace(seq_id, std::move(seq_ptr));
            ingested_count++;
        }

        if (ingested_count > 0) {
            spdlog::debug("Scheduler: Ingested {} new sequences, total running count: {}",
                         ingested_count, running_sequences_.size());
        }
    }

    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> Scheduler::select_batch() {
        spdlog::debug("Scheduler: Selecting batch from {} running sequences", running_sequences_.size());

        std::vector<uint64_t> prefill_seq_ids;
        std::vector<uint64_t> decode_seq_ids;

        // In a real implementation, we would select sequences for prefill and decode
        // based on sequence status and other criteria

        // First, count sequences by status
        size_t prefill_candidates = 0;
        size_t decode_candidates = 0;

        for (const auto& [seq_id, seq_ptr] : running_sequences_) {
            if (seq_ptr->status == sequence::SequenceStatus::PREFILLING) {
                prefill_candidates++;
            } else if (seq_ptr->status == sequence::SequenceStatus::DECODING) {
                decode_candidates++;
            }
        }

        spdlog::debug("Scheduler: Found {} prefill candidates and {} decode candidates",
                     prefill_candidates, decode_candidates);

        // In a real implementation, we would fill these vectors with sequence IDs

        spdlog::debug("Scheduler: Selected {} prefill sequences and {} decode sequences for batch",
                     prefill_seq_ids.size(), decode_seq_ids.size());

        return {prefill_seq_ids, decode_seq_ids};
    }

    bool Scheduler::allocate_pages_for_sequence(sequence::Sequence& seq) {
        uint64_t seq_id = seq.sequence_id;
        spdlog::debug("Scheduler: Allocating pages for sequence_id={}", seq_id);

        // Calculate how many *new* pages are needed based on current length and page table size
        size_t current_len = seq.get_logical_len();
        size_t current_blocks = seq.page_table.size();
        size_t required_blocks = (current_len + TOKEN_CAPACITY_PER_PAGE - 1) / TOKEN_CAPACITY_PER_PAGE;

        if (required_blocks <= current_blocks) {
            spdlog::debug("Scheduler: Sequence_id={} already has enough pages ({}) for {} tokens",
                         seq_id, current_blocks, current_len);
            return true; // Already has enough pages allocated
        }

        size_t num_pages_to_alloc = required_blocks - current_blocks;
        spdlog::debug("Scheduler: Sequence_id={} needs {} new pages (current_len={}, current_blocks={}, required_blocks={})",
                     seq_id, num_pages_to_alloc, current_len, current_blocks, required_blocks);

        // Check allocator free space before trying
        size_t free_pages_before = allocator_.get_num_free_pages();
        if (free_pages_before < num_pages_to_alloc) {
            spdlog::warn("Scheduler: Page allocation likely to fail for sequence_id={}. Need {} pages but only {} are free.",
                        seq_id, num_pages_to_alloc, free_pages_before);
        }

        // Try to allocate pages one by one
        std::vector<uint32_t> newly_allocated_pages;
        for (size_t i = 0; i < num_pages_to_alloc; ++i) {
            auto page_id_opt = allocator_.allocate_page();
            if (!page_id_opt) {
                spdlog::error("Scheduler: Page allocation failed for sequence_id={} at iteration {}/{}. Allocator out of pages.",
                             seq_id, i+1, num_pages_to_alloc);

                // Free any pages we've already allocated in this attempt
                for (uint32_t allocated_page_id : newly_allocated_pages) {
                    spdlog::debug("Scheduler: Rolling back allocation of page_id={} for sequence_id={}",
                                 allocated_page_id, seq_id);
                    allocator_.free_page(allocated_page_id);
                }

                return false;
            }

            uint32_t page_id = page_id_opt.value();
            newly_allocated_pages.push_back(page_id);
            spdlog::trace("Scheduler: Allocated page_id={} for sequence_id={} (allocation {}/{})",
                         page_id, seq_id, i+1, num_pages_to_alloc);

            seq.append_page(page_id);
        }

        size_t free_pages_after = allocator_.get_num_free_pages();
        spdlog::debug("Scheduler: Successfully allocated {} pages for sequence_id={}. Allocator free pages: {} -> {}",
                     num_pages_to_alloc, seq_id, free_pages_before, free_pages_after);
        return true;
    }


    BatchDetails Scheduler::build_batch_details(
         const std::vector<uint64_t>& prefill_seq_ids,
         const std::vector<uint64_t>& decode_seq_ids
    ) {
        spdlog::debug("Scheduler: Building batch details for {} prefill sequences and {} decode sequences",
                     prefill_seq_ids.size(), decode_seq_ids.size());

        auto start_time = std::chrono::steady_clock::now();

        BatchDetails details;
        details.num_prefill_sequences = prefill_seq_ids.size();
        details.num_decode_sequences = decode_seq_ids.size();

        // Set the attention type from scheduler configuration
        details.attention_type = attention_type_;

        spdlog::debug("Scheduler: Using attention_type={} for batch",
                    details.attention_type == AttentionType::STANDARD ? "STANDARD" : "PAGED");

        std::vector<mx::array> batch_token_ids;
        std::vector<mx::array> batch_positions;
        std::vector<mx::array> batch_block_tables; // Placeholder for consolidated table

        size_t total_tokens = 0;
        size_t max_seq_len = 0;

        auto process_sequence = [&](uint64_t seq_id, bool is_prefill) {
            auto it = running_sequences_.find(seq_id);
            if (it == running_sequences_.end()) {
                 spdlog::error("Scheduler: Sequence_id={} not found in running_sequences_ during build_batch_details", seq_id);
                 return; // Skip this sequence
            }
            sequence::Sequence& seq = *(it->second);
            spdlog::trace("Scheduler: Processing sequence_id={} for batch (is_prefill={})", seq_id, is_prefill);

            int input_len = 0;
            mx::array tokens_to_process = mx::array({});
            mx::array positions_for_tokens = mx::array({});

            // Real implementation would be used here
            // if (is_prefill) {
            //      // Simple prefill: process the whole prompt
            //      tokens_to_process = mx::array(std::vector<int32_t>(seq.tokens.begin(), seq.tokens.begin() + seq.prompt_len));
            //      input_len = seq.prompt_len;
            //      positions_for_tokens = mx::arange(0, input_len);
            //      seq.status = sequence::SequenceStatus::DECODING; // Transition state after prefill
            //      spdlog::debug("Scheduler: Building prefill batch for sequence_id={}, len={}", seq_id, input_len);
            // } else { // Decode
            //      tokens_to_process = mx::array({seq.tokens.back()}); // Process the last token
            //      input_len = 1;
            //      positions_for_tokens = mx::array({static_cast<int>(seq.get_logical_len() - 1)}); // Position is current length - 1
            //      spdlog::debug("Scheduler: Building decode batch for sequence_id={}, pos={}", seq_id, positions_for_tokens.item<int>());
            // }

            batch_token_ids.push_back(tokens_to_process);
            batch_positions.push_back(positions_for_tokens);
            details.sequence_ids.push_back(seq_id);
            details.input_lengths.push_back(input_len);

            size_t context_len = seq.get_logical_len() - input_len; // Length *before* this step
            details.context_lengths.push_back(context_len);
            max_seq_len = std::max(max_seq_len, seq.get_logical_len());

            total_tokens += input_len;

            // --- TODO: Build Consolidated Block Table ---
            // This is highly dependent on the kernel's expected format.
            // Example: Flatten page table for this sequence and add to a batch list
            // batch_block_tables.push_back(mx::array(seq.page_table));

            spdlog::trace("Scheduler: Added sequence_id={} to batch with input_len={}, context_len={}",
                         seq_id, input_len, context_len);
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

        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        spdlog::debug("Scheduler: Built batch details with {} total tokens across {} sequences in {}µs. Max sequence length: {}",
                     total_tokens, details.sequence_ids.size(), processing_time, max_seq_len);
        return details;
    }

    void Scheduler::process_batch_output(const mx::array& logits, const BatchDetails& batch_details) {
        // Logits shape: [total_tokens_in_step, vocab_size]
        spdlog::debug("Scheduler: Processing batch output for {} sequences, {} total tokens",
                     batch_details.sequence_ids.size(), batch_details.total_tokens_in_step);

        auto start_time = std::chrono::steady_clock::now();
        size_t successful_sequences = 0;
        size_t finished_sequences = 0;

        size_t current_token_offset = 0;
        for (size_t i = 0; i < batch_details.sequence_ids.size(); ++i) {
            uint64_t seq_id = batch_details.sequence_ids[i];
            int num_tokens_for_seq = batch_details.input_lengths[i];

            spdlog::trace("Scheduler: Processing output for sequence_id={} (batch position {}), {} tokens",
                         seq_id, i, num_tokens_for_seq);

            auto it = running_sequences_.find(seq_id);
            if (it == running_sequences_.end()) {
                spdlog::error("Scheduler: Sequence_id={} from batch not found in running_sequences_ during output processing",
                             seq_id);
                current_token_offset += num_tokens_for_seq;
                continue;
            }

            sequence::Sequence& seq = *(it->second);

            // Starting sequence state
            auto orig_status = seq.status;
            size_t orig_len = seq.get_logical_len();
            spdlog::trace("Scheduler: Sequence_id={} processing start - status={}, logical_len={}",
                         seq_id, static_cast<int>(orig_status), orig_len);

            // 1. Extract the relevant slice of logits for the *last* token of this sequence
            //    For prefill, we only care about the logit for the token *after* the prompt.
            //    For decode, there's only one token's logit.
            size_t logit_index = current_token_offset + num_tokens_for_seq - 1;
            mx::array seq_logits = mx::slice(
                    logits,
                    {(int)logit_index, 0},
                    {(int)logit_index + 1, logits.shape(1)});
            spdlog::trace("Scheduler: Extracted logits for sequence_id={} at offset {}",
                         seq_id, logit_index);

            // 2. Apply Logit Processors
            spdlog::trace("Scheduler: Applying logit processors for sequence_id={}", seq_id);
            try {
                std::vector<std::unique_ptr<logit_processors::ILogitProcessor>> processors =
                    logit_processors::create_processors(seq.logits_params);

                if (!processors.empty()) {
                    spdlog::trace("Scheduler: Created {} logit processors for sequence_id={}",
                                 processors.size(), seq_id);
                }

                for(size_t proc_idx = 0; proc_idx < processors.size(); proc_idx++) {
                    const auto& processor = processors[proc_idx];
                    spdlog::trace("Scheduler: Applying logit processor #{} for sequence_id={}",
                                 proc_idx+1, seq_id);
                    seq_logits = processor->process_logits(seq_logits, seq.logits_params, seq);
                }

                spdlog::trace("Scheduler: Completed logit processing for sequence_id={}", seq_id);
            } catch (const std::exception& e) {
                spdlog::error("Scheduler: Error in logit processing for sequence_id={}: {}", seq_id, e.what());
                current_token_offset += num_tokens_for_seq;
                continue;
            }

            // 3. Sample Next Token
            spdlog::trace("Scheduler: Sampling next token for sequence_id={}", seq_id);
            int32_t next_token_id;
            try {
                std::unique_ptr<samplers::ISampler> sampler = samplers::create_sampler(seq.sampling_params);
                mx::array next_token_id_array = sampler->next_token(seq_logits, seq.sampling_params, rng_);
                next_token_id = next_token_id_array.item<int32_t>(); // Assuming sampler returns scalar array
                spdlog::debug("Scheduler: Sampled token_id={} for sequence_id={}", next_token_id, seq_id);
            } catch (const std::exception& e) {
                spdlog::error("Scheduler: Error sampling next token for sequence_id={}: {}", seq_id, e.what());
                current_token_offset += num_tokens_for_seq;
                continue;
            }

            // 4. Update Sequence State - append token
            try {
                spdlog::trace("Scheduler: Appending token_id={} to sequence_id={}", next_token_id, seq_id);
                seq.append_token(next_token_id);
                spdlog::trace("Scheduler: Successfully appended token_id={} to sequence_id={}, new length={}",
                             next_token_id, seq_id, seq.get_logical_len());
            } catch (const std::exception& e) {
                spdlog::error("Scheduler: Failed to append token to sequence_id={}: {}", seq_id, e.what());
                current_token_offset += num_tokens_for_seq;
                continue;
            }

            // 5. Check Stop Conditions
            bool finished = false;
            sequence::FinishReason reason = sequence::FinishReason::STOP; // Default assumption

            if (seq.get_generation_len() >= static_cast<size_t>(seq.stop_criteria.max_generated_tokens)) {
                 finished = true;
                 reason = sequence::FinishReason::LENGTH;
                 spdlog::info("Scheduler: Sequence_id={} finished due to reaching max length ({} tokens)",
                             seq_id, seq.get_generation_len());
            } else {
                 // Check stop tokens
                 for (int32_t stop_id : seq.stop_criteria.stop_token_ids) {
                      if (next_token_id == stop_id) {
                           finished = true;
                           reason = sequence::FinishReason::STOP;
                           spdlog::info("Scheduler: Sequence_id={} finished due to stop token {}",
                                       seq_id, stop_id);
                           break;
                      }
                 }
                 // TODO: Check for other stop reasons (user sequence, tool use signal)
            }

            // 6. Send token to postprocessor
            std::unique_ptr<PostprocessingData> pp_data = std::make_unique<PostprocessingData>();
            pp_data->request_id = seq.sequence_id;
            pp_data->next_token_id = next_token_id;
            pp_data->is_final_delta = finished;
            pp_data->finish_reason = reason;

            // TODO: Populate logprobs if requested
            // pp_data->top_logprobs = ...

            spdlog::debug("Scheduler: Sending token to postprocessor for sequence_id={}, token_id={}, is_final={}",
                         seq_id, next_token_id, finished);

            if (postprocessing_queue_.push(std::move(pp_data))) {
                spdlog::debug("Scheduler: Successfully sent token to postprocessor for sequence_id={}", seq_id);
                successful_sequences++;
            } else {
                spdlog::error("Scheduler: Failed to send token to postprocessor for sequence_id={} (queue full)",
                             seq_id);
                // Mark sequence as errored
                seq.status = sequence::SequenceStatus::ERROR;
                finished = true; // Mark as finished to trigger cleanup

                // Fallback to direct response writing
                try {
                    ipc::ResponseDeltaSlot delta;
                    delta.request_id = seq.sequence_id;
                    delta.num_tokens_in_delta = 1;
                    delta.tokens[0] = next_token_id;
                    delta.is_final_delta = finished;
                    delta.finish_reason = reason;

                    spdlog::warn("Scheduler: Fallback - Direct response writing for sequence_id={}", seq_id);
                    response_writer_.write_delta(delta);
                } catch (const ipc::ResponseWriterError& e) {
                    spdlog::error("Scheduler: Fallback failed - Could not write response for sequence_id={}: {}",
                                 seq_id, e.what());
                }
            }

            // 7. Update sequence status if finished
            if (finished) {
                if (seq.status != sequence::SequenceStatus::ERROR) {
                    seq.status = sequence::SequenceStatus::COMPLETED;
                }
                finished_sequences++;

                spdlog::info("Scheduler: Marked sequence_id={} as finished with status={}, reason={}, generated {} tokens",
                           seq_id, static_cast<int>(seq.status), static_cast<int>(reason), seq.get_generation_len());
            } else {
                spdlog::trace("Scheduler: Sequence_id={} continues with status={}, tokens={}, generation_len={}",
                             seq_id, static_cast<int>(seq.status), seq.get_logical_len(), seq.get_generation_len());
            }

            // Move offset for the next sequence in the batch
            current_token_offset += num_tokens_for_seq;
        }

        // Calculate processing times
        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time).count();

        double avg_time_per_seq = (batch_details.sequence_ids.size() > 0) ?
            static_cast<double>(processing_time) / batch_details.sequence_ids.size() : 0.0;

        spdlog::debug("Scheduler: Processed batch output in {}µs ({:.2f}µs/seq). Success: {}/{}, Finished: {}",
                     processing_time, avg_time_per_seq, successful_sequences, batch_details.sequence_ids.size(),
                     finished_sequences);
    }

    void Scheduler::free_sequence_pages(const sequence::Sequence& seq) {
        uint64_t seq_id = seq.sequence_id;
        size_t page_count = seq.page_table.size();

        if (page_count == 0) {
            spdlog::debug("Scheduler: No pages to free for sequence_id={}", seq_id);
            return;
        }

        spdlog::debug("Scheduler: Freeing {} pages for sequence_id={}", page_count, seq_id);

        size_t free_pages_before = allocator_.get_num_free_pages();
        size_t freed_count = 0;

        for (uint32_t page_id : seq.page_table) {
            spdlog::trace("Scheduler: Freeing page_id={} for sequence_id={} ({}/{})",
                         page_id, seq_id, freed_count+1, page_count);
            try {
                allocator_.free_page(page_id);
                freed_count++;
            } catch (const std::exception& e) {
                spdlog::error("Scheduler: Error freeing page_id={} for sequence_id={}: {}",
                             page_id, seq_id, e.what());
            }
        }

        size_t free_pages_after = allocator_.get_num_free_pages();
        spdlog::debug("Scheduler: Freed {}/{} pages for sequence_id={}. Allocator free pages: {} -> {}",
                     freed_count, page_count, seq_id, free_pages_before, free_pages_after);
    }

    void Scheduler::cleanup_finished_sequences() {
        spdlog::debug("Scheduler: Running cleanup for finished sequences, current count: {}",
                     running_sequences_.size());

        size_t removed_count = 0;

        auto it = running_sequences_.begin();
        while (it != running_sequences_.end()) {
             sequence::Sequence& seq = *(it->second);
             uint64_t seq_id = seq.sequence_id;

             bool should_remove =
                 seq.status == sequence::SequenceStatus::COMPLETED ||
                 seq.status == sequence::SequenceStatus::ERROR ||
                 seq.cancelled.load(std::memory_order_acquire);

             if (should_remove) {
                  sequence::SequenceStatus status = seq.status;
                  bool was_cancelled = seq.cancelled.load(std::memory_order_acquire);
                  size_t prompt_len = seq.prompt_len;
                  size_t gen_len = seq.get_generation_len();

                  spdlog::info("Scheduler: Cleaning up sequence_id={}: status={}, cancelled={}, prompt_len={}, gen_len={}",
                              seq_id, static_cast<int>(status), was_cancelled, prompt_len, gen_len);

                  free_sequence_pages(seq);
                  it = running_sequences_.erase(it); // Remove from running map
                  removed_count++;

                  spdlog::debug("Scheduler: Successfully removed sequence_id={} from running sequences", seq_id);
             } else {
                  spdlog::trace("Scheduler: Keeping sequence_id={} with status={}",
                               seq_id, static_cast<int>(seq.status));
                  ++it;
             }
        }

        if (removed_count > 0) {
            spdlog::info("Scheduler: Cleanup removed {} finished sequences, {} sequences remaining",
                       removed_count, running_sequences_.size());
        } else {
            spdlog::debug("Scheduler: No sequences to clean up (all {} are still active)",
                        running_sequences_.size());
        }
    }

} // namespace pie_core::engine
