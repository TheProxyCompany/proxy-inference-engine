#pragma once

#include "engine/raw_request.hpp"
#include "sequence/sequence.hpp"
#include "ipc/shared_memory_manager.hpp"
#include <boost/lockfree/spsc_queue.hpp>
#include <tokenizers_cpp.h>

#include <memory>
#include <thread>
#include <atomic>
#include <string>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

namespace pie_core::engine {

    class RequestPreprocessor {
    public:
        using RawRequestQueue = boost::lockfree::spsc_queue<
            std::unique_ptr<RawRequestData>,
            boost::lockfree::capacity<1024>
        >;
        using ProcessedSequenceQueue = boost::lockfree::spsc_queue<
            std::unique_ptr<sequence::Sequence>,
            boost::lockfree::capacity<1024>
        >;

        /**
         * @brief Constructor.
         * @param input_raw_requests_queue Reference to the SPSC queue from RequestReader.
         * @param output_sequences_queue Reference to the SPSC queue for the Scheduler.
         * @param shm_manager Shared memory manager for prompt access.
         * @param model_path Path to the tokenizer model/config files.
         */
        RequestPreprocessor(
            RawRequestQueue& input_raw_requests_queue,
            ProcessedSequenceQueue& output_sequences_queue,
            ipc::SharedMemoryManager& shm_manager,
            const std::string& model_path
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
        void run_loop();

        RawRequestQueue& input_queue_;
        ProcessedSequenceQueue& output_queue_;
        ipc::SharedMemoryManager& shm_manager_;
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

        std::atomic<bool> stop_flag_{false};
        std::thread worker_thread_;
    };

} // namespace pie_core::engine
