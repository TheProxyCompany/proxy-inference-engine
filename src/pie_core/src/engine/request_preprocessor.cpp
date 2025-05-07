#include "engine/request_preprocessor.hpp"
#include "engine/raw_request.hpp"
#include "sequence/sequence.hpp"
#include "ipc/shared_memory_manager.hpp"
#include <tokenizers_cpp.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>
#include "utils/read_file.hpp"

namespace pie_core::engine {

    namespace fs = std::filesystem;

    RequestPreprocessor::RequestPreprocessor(
        RawRequestQueue& input_raw_requests_queue,
        ProcessedSequenceQueue& output_sequences_queue,
        ipc::SharedMemoryManager& shm_manager,
        const std::string& model_path
    ):
        input_queue_(input_raw_requests_queue),
        output_queue_(output_sequences_queue),
        shm_manager_(shm_manager),
        stop_flag_{false}
    {
        spdlog::info("RequestPreprocessor: Initializing with model_path='{}'", model_path);

        fs::path model_directory = fs::path(model_path);
        if (!fs::exists(model_directory)) {
            spdlog::critical("RequestPreprocessor: Model directory does not exist: {}", model_directory.string());
            throw std::runtime_error("Model directory does not exist: " + model_directory.string());
        }
        spdlog::debug("RequestPreprocessor: Model directory exists at: {}", model_directory.string());

        // Load tokenizer from model directory
        spdlog::info("RequestPreprocessor: Attempting to load tokenizer from model directory");

        if (fs::exists(model_directory / "tokenizer.json")) {
            const auto tokenizer_file_path = (model_directory / "tokenizer.json").string();
            spdlog::debug("RequestPreprocessor: Found tokenizer.json at: {}", tokenizer_file_path);

            try {
                auto file_contents_blob = load_file_bytes(tokenizer_file_path);
                spdlog::debug("RequestPreprocessor: Loaded tokenizer.json file, size: {} bytes", file_contents_blob.size());

                tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(file_contents_blob);
                spdlog::info("RequestPreprocessor: Successfully initialized JSON tokenizer from '{}'", tokenizer_file_path);
            } catch (const std::exception& e) {
                spdlog::critical("RequestPreprocessor: Failed to load JSON tokenizer from '{}': {}",
                               tokenizer_file_path, e.what());
                throw;
            }
        } else if (fs::exists(model_directory / "tokenizer.model")) {
            const auto tokenizer_file_path = (model_directory / "tokenizer.model").string();
            spdlog::debug("RequestPreprocessor: Found tokenizer.model at: {}", tokenizer_file_path);

            try {
                auto file_contents_blob = load_file_bytes(tokenizer_file_path);
                spdlog::debug("RequestPreprocessor: Loaded tokenizer.model file, size: {} bytes", file_contents_blob.size());

                tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(file_contents_blob);
                spdlog::info("RequestPreprocessor: Successfully initialized SentencePiece tokenizer from '{}'", tokenizer_file_path);
            } catch (const std::exception& e) {
                spdlog::critical("RequestPreprocessor: Failed to load SentencePiece tokenizer from '{}': {}",
                               tokenizer_file_path, e.what());
                throw;
            }
        } else {
            spdlog::critical("RequestPreprocessor: No tokenizer.json or tokenizer.model found in {}", model_directory.string());
            throw std::runtime_error("No tokenizer.json or tokenizer.model found in " + model_directory.string());
        }

        spdlog::info("RequestPreprocessor: Initialization complete");
    }

    RequestPreprocessor::~RequestPreprocessor() {
        spdlog::info("RequestPreprocessor: Destructor called");
        stop();
        spdlog::debug("RequestPreprocessor: Destructor complete");
    }

    void RequestPreprocessor::stop() {
        bool was_already_stopping = stop_flag_.exchange(true, std::memory_order_acq_rel);
        if (!was_already_stopping) {
            spdlog::info("RequestPreprocessor: Stop signal received");
        } else {
            spdlog::debug("RequestPreprocessor: Duplicate stop signal received (already stopping)");
        }
    }

    void RequestPreprocessor::run_loop() {
        spdlog::info("RequestPreprocessor: Run loop entered");
        std::unique_ptr<RawRequestData> raw_request_ptr;
        uint64_t loop_counter = 0;
        uint64_t requests_processed = 0;
        uint64_t sequences_enqueued = 0;

        while (!stop_flag_.load(std::memory_order_acquire)) {
            loop_counter++;

            // Log occasional statistics at trace level
            if (loop_counter % 1000 == 0) {
                spdlog::trace("RequestPreprocessor: Run loop iteration {}, processed {} requests, enqueued {} sequences",
                             loop_counter, requests_processed, sequences_enqueued);
            }

            // Try to get a request from the input queue
            if (input_queue_.pop(raw_request_ptr)) {
                requests_processed++;
                uint64_t request_id = raw_request_ptr->request_id;
                spdlog::info("RequestPreprocessor: Processing request_id={} (request #{} in this session)",
                            request_id, requests_processed);

                // Track timing for this request's processing
                auto start_time = std::chrono::steady_clock::now();
                std::string prompt_payload = raw_request_ptr->prompt_payload;

                spdlog::debug("RequestPreprocessor: Request_id={} has prompt size {} bytes, temp={}, top_p={}, max_tokens={}",
                            request_id,
                            prompt_payload.size(),
                            raw_request_ptr->sampling_params.temperature,
                            raw_request_ptr->sampling_params.top_p,
                            raw_request_ptr->stop_criteria.max_generated_tokens);

                // --- 0. Chat Template Application ---
                if (raw_request_ptr->type == PromptType::CHAT_HISTORY) {
                    // --- TODO: Chat Templating Logic ---
                    // 1. Parse prompt_payload (if it's e.g., a JSON string of turns)
                    // 2. Apply chat template using stored chat_template_str_ and special_tokens_map_
                    //    (This is where having a C++ Jinja-like mini-engine or robust string formatting is needed,
                    //     or enhancing your tokenizers-cpp fork to expose Rust's chat template application)
                    // prompt_payload = apply_chat_template(...);
                    spdlog::debug("RequestPreprocessor: Applying chat template for request_id={}", request_id);
                }

                // --- 1. Tokenization ---
                std::vector<int32_t> token_ids = {0}; // Placeholder token

                // Tokenization logic (commented out but with improved logging for when it's implemented)
                // try {
                //     auto tokenization_start = std::chrono::steady_clock::now();
                //     spdlog::debug("RequestPreprocessor: Starting tokenization for request_id={}", request_id);
                //
                //     token_ids = tokenizer_->Encode(prompt_payload);
                //
                //     auto tokenization_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                //         std::chrono::steady_clock::now() - tokenization_start).count();
                //
                //     if (token_ids.empty()) {
                //         spdlog::error("RequestPreprocessor: Tokenization failed for request_id={} - empty token list", request_id);
                //         continue;
                //     }
                //
                //     spdlog::info("RequestPreprocessor: Tokenized request_id={}, num_tokens={} in {}ms",
                //                 request_id, token_ids.size(), tokenization_duration);
                //
                //     // Log token IDs at trace level (truncate if very long)
                //     if (token_ids.size() <= 20) {
                //         std::string token_str;
                //         for (size_t i = 0; i < token_ids.size(); i++) {
                //             if (i > 0) token_str += ",";
                //             token_str += std::to_string(token_ids[i]);
                //         }
                //         spdlog::trace("RequestPreprocessor: Token IDs for request_id={}: [{}]", request_id, token_str);
                //     } else {
                //         std::string token_start;
                //         for (size_t i = 0; i < 10; i++) {
                //             if (i > 0) token_start += ",";
                //             token_start += std::to_string(token_ids[i]);
                //         }
                //         std::string token_end;
                //         for (size_t i = token_ids.size() - 10; i < token_ids.size(); i++) {
                //             if (i > token_ids.size() - 10) token_end += ",";
                //             token_end += std::to_string(token_ids[i]);
                //         }
                //         spdlog::trace("RequestPreprocessor: Token IDs for request_id={}: [{},... ({} more),{}]",
                //                      request_id, token_start, token_ids.size() - 20, token_end);
                //     }
                // } catch (const std::exception& e) {
                //     spdlog::error("RequestPreprocessor: Exception during tokenization for request_id={}: {}",
                //                  request_id, e.what());
                //     continue;
                // }

                // --- 2. Deallocate Raw Prompt String from SHM ---
                if (raw_request_ptr->_shm_prompt_size > 0) {
                    try {
                        void* bulk_shm_base = shm_manager_.get_segment_base_address();
                        if (bulk_shm_base) {
                            void* shm_block_to_free = static_cast<char*>(bulk_shm_base) + raw_request_ptr->_shm_prompt_offset;
                            spdlog::debug("RequestPreprocessor: Deallocating SHM for prompt of request_id={} at offset {}, size {}",
                                        request_id, raw_request_ptr->_shm_prompt_offset, raw_request_ptr->_shm_prompt_size);
                            shm_manager_.deallocate(shm_block_to_free);
                            spdlog::debug("RequestPreprocessor: Successfully deallocated SHM for prompt of request_id={}", request_id);
                        } else {
                            spdlog::error("RequestPreprocessor: Failed to get bulk SHM base address for deallocation, request_id={}. Leaking SHM block.",
                                         request_id);
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("RequestPreprocessor: Exception deallocating SHM for prompt of request_id={}: {}",
                                     request_id, e.what());
                    }
                }

                // --- 3. Construct Sequence Object ---
                spdlog::debug("RequestPreprocessor: Creating Sequence object for request_id={}", request_id);
                std::unique_ptr<sequence::Sequence> sequence_ptr;
                try {
                    sequence_ptr = std::make_unique<sequence::Sequence>(
                        request_id,
                        sequence::SequenceStatus::WAITING,
                        raw_request_ptr->arrival_timestamp_ns,
                        token_ids,
                        token_ids.size(),
                        raw_request_ptr->sampling_params,
                        raw_request_ptr->logits_params,
                        raw_request_ptr->stop_criteria,
                        raw_request_ptr->ipc_handles
                    );
                    spdlog::debug("RequestPreprocessor: Successfully created Sequence object for request_id={}", request_id);
                } catch (const std::exception& e) {
                    spdlog::error("RequestPreprocessor: Failed to create Sequence object for request_id={}: {}",
                                 request_id, e.what());
                    continue;
                }

                // --- 4. Enqueue Processed Sequence for Scheduler ---
                spdlog::debug("RequestPreprocessor: Attempting to enqueue Sequence for request_id={} to scheduler", request_id);

                // Push to output queue
                if (!output_queue_.push(std::move(sequence_ptr))) {
                    spdlog::critical("RequestPreprocessor: Failed to enqueue Sequence for request_id={} to scheduler queue (FULL!). Sequence dropped.",
                                    request_id);
                } else {
                    sequences_enqueued++;

                    // Calculate processing time
                    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count();

                    spdlog::info("RequestPreprocessor: Successfully enqueued Sequence for request_id={} to scheduler in {}ms",
                                request_id, processing_time);
                }

            } else {
                // No request in queue, check stop flag
                if (stop_flag_.load(std::memory_order_acquire)) {
                    spdlog::debug("RequestPreprocessor: Stop flag detected, exiting run loop");
                    break;
                }

                // Sleep briefly to prevent busy-waiting when queue is empty
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        spdlog::info("RequestPreprocessor: Run loop exited after {} iterations. Processed {} requests, enqueued {} sequences.",
                   loop_counter, requests_processed, sequences_enqueued);
    }

} // namespace pie_core::engine
