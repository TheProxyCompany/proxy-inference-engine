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
        fs::path model_directory = fs::path(model_path);
        if (!fs::exists(model_directory)) {
            throw std::runtime_error("Model directory does not exist: " + model_directory.string());
        }

        if (fs::exists(model_directory / "tokenizer.json")) {
            const auto tokenizer_file_path = (model_directory / "tokenizer.json").string();
            auto file_contents_blob = load_file_bytes(tokenizer_file_path);
            tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(file_contents_blob);
        } else if (fs::exists(model_directory / "tokenizer.model")) {
            const auto tokenizer_file_path = (model_directory / "tokenizer.model").string();
            auto file_contents_blob = load_file_bytes(tokenizer_file_path);
            tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(file_contents_blob);
        } else {
            throw std::runtime_error("No tokenizer.json or tokenizer.model found in " + model_directory.string());
        }

        spdlog::info("RequestPreprocessor: Tokenizer loaded successfully from '{}'.", model_directory.string());
        spdlog::info("RequestPreprocessor constructed.");
    }

    RequestPreprocessor::~RequestPreprocessor() {
        if (worker_thread_.joinable()) {
            stop_and_join();
        }
        spdlog::info("RequestPreprocessor destructed.");
    }

    void RequestPreprocessor::start() {
        if (worker_thread_.joinable()) {
            spdlog::warn("RequestPreprocessor: Start called on an already running preprocessor.");
            return;
        }
        stop_flag_.store(false, std::memory_order_relaxed);
        worker_thread_ = std::thread(&RequestPreprocessor::run_loop, this);
        spdlog::info("RequestPreprocessor: Worker thread started.");
    }

    void RequestPreprocessor::stop_and_join() {
        stop_flag_.store(true, std::memory_order_release);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
            spdlog::info("RequestPreprocessor: Worker thread joined.");
        } else {
            spdlog::info("RequestPreprocessor: Stop called, but worker thread was not joinable (already joined or not started).");
        }
    }

    void RequestPreprocessor::run_loop() {
        spdlog::info("RequestPreprocessor: Run loop entered.");
        std::unique_ptr<RawRequestData> raw_request_ptr;

        while (!stop_flag_.load(std::memory_order_acquire)) {
            if (input_queue_.pop(raw_request_ptr)) {
                spdlog::debug("RequestPreprocessor: Processing request ID {}.", raw_request_ptr->request_id);
                std::vector<int32_t> token_ids;
                std::string prompt_payload = raw_request_ptr->prompt_payload;

                // --- 0. Chat Template Application ---
                if (raw_request_ptr->type == PromptType::CHAT_HISTORY) {
                    // --- TODO: Chat Templating Logic ---
                    // 1. Parse prompt_payload (if it's e.g., a JSON string of turns)
                    // 2. Apply chat template using stored chat_template_str_ and special_tokens_map_
                    //    (This is where having a C++ Jinja-like mini-engine or robust string formatting is needed,
                    //     or enhancing your tokenizers-cpp fork to expose Rust's chat template application)
                    // prompt_payload = apply_chat_template(...);
                    spdlog::debug("RequestPreprocessor: Applying chat template for request ID {}.", raw_request_ptr->request_id);
                }

                // --- 1. Tokenization ---
                try {
                    token_ids = tokenizer_->Encode(prompt_payload);
                    if (token_ids.empty()) {
                        spdlog::error("RequestPreprocessor: Tokenization failed for request ID {}.", raw_request_ptr->request_id);
                        continue;
                    }
                    spdlog::debug("RequestPreprocessor: Tokenized request ID {}, num_tokens: {}", raw_request_ptr->request_id, token_ids.size());

                } catch (const std::exception& e) {
                    spdlog::error("RequestPreprocessor: Exception during tokenization for request ID {}: {}", raw_request_ptr->request_id, e.what());
                    continue;
                }

                // --- 2. Deallocate Raw Prompt String from SHM ---
                if (raw_request_ptr->_shm_prompt_size > 0) {
                    try {
                        void* bulk_shm_base = shm_manager_.get_segment_base_address();
                        if (bulk_shm_base) {
                            void* shm_block_to_free = static_cast<char*>(bulk_shm_base) + raw_request_ptr->_shm_prompt_offset;
                            shm_manager_.deallocate(shm_block_to_free);
                            spdlog::debug("RequestPreprocessor: Deallocated SHM for prompt of request ID {}.", raw_request_ptr->request_id);
                        } else {
                            spdlog::error("RequestPreprocessor: Failed to get bulk SHM base address for deallocation, request ID {}. Leaking SHM block.", raw_request_ptr->request_id);
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("RequestPreprocessor: Exception deallocating SHM for prompt of request ID {}: {}", raw_request_ptr->request_id, e.what());
                    }
                }

                // --- 3. Construct Sequence Object ---
                std::unique_ptr<sequence::Sequence> sequence_ptr;
                try {
                    sequence_ptr = std::make_unique<sequence::Sequence>(
                        raw_request_ptr->request_id,
                        sequence::SequenceStatus::WAITING,
                        raw_request_ptr->arrival_timestamp_ns,
                        token_ids,
                        token_ids.size(),
                        raw_request_ptr->sampling_params,
                        raw_request_ptr->logits_params,
                        raw_request_ptr->stop_criteria,
                        raw_request_ptr->ipc_handles
                    );
                } catch (const std::exception& e) {
                    spdlog::error("RequestPreprocessor: Failed to create Sequence object for request ID {}: {}", raw_request_ptr->request_id, e.what());
                    continue;
                }

                // --- 4. Enqueue Processed Sequence for Scheduler ---
                if (!output_queue_.push(std::move(sequence_ptr))) {
                    spdlog::critical("RequestPreprocessor: Failed to enqueue processed sequence ID {} to scheduler queue (FULL!). Sequence dropped.", raw_request_ptr->request_id);
                } else {
                    spdlog::debug("RequestPreprocessor: Enqueued sequence ID {} for scheduler.", raw_request_ptr->request_id);
                }

            } else {
                if (stop_flag_.load(std::memory_order_acquire)) {
                    break;
                }
                std::this_thread::yield();
            }
        }
        spdlog::info("RequestPreprocessor: Run loop exited.");
    }

} // namespace pie_core::engine
