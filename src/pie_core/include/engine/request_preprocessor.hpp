#pragma once

#include "engine/raw_request.hpp"
#include "sequence/sequence.hpp"
#include "ipc/ipc_reader.hpp"

#include <memory>
#include <thread>
#include <atomic>
#include <string>

namespace pie_core::engine {

    class RequestPreprocessor {
    public:
        // Define the types for the input and output SPSC queues
        using RawRequestQueue = ipc::SPSCQueue<std::unique_ptr<RawRequestData>>;
        using ProcessedSequenceQueue = ipc::SPSCQueue<std::unique_ptr<sequence::Sequence>>;

        /**
         * @brief Constructor.
         * @param input_raw_requests_queue Reference to the SPSC queue from IPCReader.
         * @param output_sequences_queue Reference to the SPSC queue for the Scheduler.
         * @param tokenizer_path Path to the tokenizer model/config files.
         */
        RequestPreprocessor(
            RawRequestQueue& input_raw_requests_queue,
            ProcessedSequenceQueue& output_sequences_queue,
            const std::string& tokenizer_path // Or pass a pre-initialized Tokenizer unique_ptr
        );

        ~RequestPreprocessor();

        /**
         * @brief Starts the preprocessor thread.
         */
        void start();

        /**
         * @brief Signals the preprocessor thread to stop and joins it.
         */
        void stop_and_join();

        // Prevent copying and moving
        RequestPreprocessor(const RequestPreprocessor&) = delete;
        RequestPreprocessor& operator=(const RequestPreprocessor&) = delete;
        RequestPreprocessor(RequestPreprocessor&&) = delete;
        RequestPreprocessor& operator=(RequestPreprocessor&&) = delete;

    private:
        void run_loop(); // The actual processing loop run by the thread

        RawRequestQueue& input_queue_;
        ProcessedSequenceQueue& output_queue_;
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_; // Owns the tokenizer

        std::atomic<bool> stop_flag_{false};
        std::thread worker_thread_;
    };

} // namespace pie_core::engine
