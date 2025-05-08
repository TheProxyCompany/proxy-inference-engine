#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include "ipc/response_writer.hpp"
#include "tokenizers/tokenizer.hpp"
#include "sequence/sequence.hpp"
#include <boost/lockfree/spsc_queue.hpp>

namespace pie_core::engine {

    /**
     * @brief Structure containing token data to be postprocessed
     */
    struct PostprocessingData {
        uint64_t request_id;
        int32_t next_token_id;
        // Optional: if logprobs needed
        // std::vector<std::pair<int32_t, float>> top_logprobs;
        bool is_final_delta;
        sequence::FinishReason finish_reason;
    };

    /**
     * @brief Single-Producer/Single-Consumer queue for token postprocessing
     */
    using PostprocessingQueue = boost::lockfree::spsc_queue<
        std::unique_ptr<PostprocessingData>,
        boost::lockfree::capacity<1024>
    >;

    /**
     * @brief Handles detokenization and IPC response writing in dedicated thread
     */
    class ResponsePostprocessor {
    public:
        /**
         * @brief Constructor
         * @param input_queue Reference to queue from which to consume tokens
         * @param response_writer Reference to ResponseWriter for sending responses via IPC
         * @param tokenizer Reference to Tokenizer for detokenization
         */
        ResponsePostprocessor(
            PostprocessingQueue& input_queue,
            ipc::ResponseWriter& response_writer,
            tokenizers::Tokenizer& tokenizer
        );

        /**
         * @brief Runs the main postprocessor loop. Will be called by Engine in its own thread.
         */
        void run_loop();

        /**
         * @brief Signals the postprocessor to stop.
         */
        void stop();

        // Prevent copying and moving
        ResponsePostprocessor(const ResponsePostprocessor&) = delete;
        ResponsePostprocessor& operator=(const ResponsePostprocessor&) = delete;
        ResponsePostprocessor(ResponsePostprocessor&&) = delete;
        ResponsePostprocessor& operator=(ResponsePostprocessor&&) = delete;

    private:
        PostprocessingQueue& input_queue_;
        ipc::ResponseWriter& response_writer_;
        tokenizers::Tokenizer& tokenizer_;

        std::atomic<bool> stop_flag_{false};
    };

} // namespace pie_core::engine
