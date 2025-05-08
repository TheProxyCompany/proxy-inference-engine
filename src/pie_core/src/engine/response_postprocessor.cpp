#include "engine/response_postprocessor.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <string>
#include <cstring>

namespace pie_core::engine {

    ResponsePostprocessor::ResponsePostprocessor(
        PostprocessingQueue& input_queue,
        ipc::ResponseWriter& response_writer,
        tokenizers::Tokenizer& tokenizer
    ):
        input_queue_(input_queue),
        response_writer_(response_writer),
        tokenizer_(tokenizer),
        stop_flag_{false}
    {
        spdlog::info("ResponsePostprocessor: Initialized");
    }

    void ResponsePostprocessor::stop() {
        bool was_already_stopping = stop_flag_.exchange(true, std::memory_order_acq_rel);
        if (!was_already_stopping) {
            spdlog::info("ResponsePostprocessor: Stop signal received");
        } else {
            spdlog::debug("ResponsePostprocessor: Duplicate stop signal received (already stopping)");
        }
    }

    void ResponsePostprocessor::run_loop() {
        spdlog::info("ResponsePostprocessor: Run loop entered");

        // Track statistics
        uint64_t loop_counter = 0;
        uint64_t tokens_processed = 0;

        while (!stop_flag_.load(std::memory_order_acquire)) {
            loop_counter++;

            // Log occasional statistics at trace level
            if (loop_counter % 1000 == 0) {
                spdlog::trace("ResponsePostprocessor: Run loop iteration {}, processed {} tokens",
                             loop_counter, tokens_processed);
            }

            // Try to get a token from the input queue
            std::unique_ptr<PostprocessingData> data_ptr;
            if (input_queue_.pop(data_ptr)) {
                tokens_processed++;
                uint64_t request_id = data_ptr->request_id;

                spdlog::debug("ResponsePostprocessor: Processing token_id={} for request_id={}",
                            data_ptr->next_token_id, request_id);

                try {
                    // 1. Detokenize the token ID
                    std::string decoded_content;
                    try {
                        decoded_content = tokenizer_.decode({data_ptr->next_token_id});
                        spdlog::debug("ResponsePostprocessor: Decoded token_id={} to: '{}'",
                                    data_ptr->next_token_id, decoded_content);
                    } catch (const tokenizers::TokenizerException& e) {
                        spdlog::error("ResponsePostprocessor: Tokenizer failed to decode token_id={} for request_id={}: {}",
                                     data_ptr->next_token_id, request_id, e.what());
                        decoded_content = "<?>";  // Placeholder for failed decoding
                    }

                    // 2. Construct ResponseDeltaSlot
                    ipc::ResponseDeltaSlot delta;
                    delta.request_id = request_id;
                    delta.num_tokens_in_delta = 1;
                    delta.tokens[0] = data_ptr->next_token_id;
                    delta.is_final_delta = data_ptr->is_final_delta;
                    delta.finish_reason = data_ptr->finish_reason;

                    // Copy content string into fixed-size buffer
                    size_t content_len = decoded_content.size();

                    // Check if content will fit in buffer
                    if (content_len >= ipc::MAX_CONTENT_BYTES) {
                        spdlog::warn("ResponsePostprocessor: Content for token_id={} exceeds max size ({} > {}). Truncating.",
                                    data_ptr->next_token_id, content_len, ipc::MAX_CONTENT_BYTES-1);
                        content_len = ipc::MAX_CONTENT_BYTES - 1;  // Leave space for null terminator
                    }

                    // Copy content and null-terminate
                    std::memcpy(delta.content, decoded_content.c_str(), content_len);
                    delta.content[content_len] = '\0';
                    delta.content_len = content_len;

                    if (data_ptr->is_final_delta) {
                        spdlog::info("ResponsePostprocessor: Sending final delta for request_id={} with finish_reason={}",
                                    request_id, static_cast<int>(data_ptr->finish_reason));
                    }

                    // 3. Write delta to IPC
                    response_writer_.write_delta(delta);
                    spdlog::debug("ResponsePostprocessor: Successfully sent delta with content '{}' for request_id={}",
                                 delta.content, request_id);

                } catch (const ipc::ResponseWriterError& e) {
                    spdlog::error("ResponsePostprocessor: Failed to write response for request_id={}: {}",
                                 request_id, e.what());
                } catch (const std::exception& e) {
                    spdlog::error("ResponsePostprocessor: Unexpected error processing token for request_id={}: {}",
                                 request_id, e.what());
                }
            } else {
                // No data in queue, check stop flag once more
                if (stop_flag_.load(std::memory_order_acquire)) {
                    spdlog::debug("ResponsePostprocessor: Stop flag detected, exiting run loop");
                    break;
                }

                // Sleep briefly to prevent busy-waiting when queue is empty
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        spdlog::info("ResponsePostprocessor: Run loop exited after {} iterations. Processed {} tokens.",
                   loop_counter, tokens_processed);
    }

} // namespace pie_core::engine
