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
         * @brief Runs the main preprocessor loop. Will be called by Engine in its own thread.
         */
        void run_loop();

        /**
         * @brief Signals the preprocessor to stop.
         */
        void stop();

        // Prevent copying and moving
        RequestPreprocessor(const RequestPreprocessor&) = delete;
        RequestPreprocessor& operator=(const RequestPreprocessor&) = delete;
        RequestPreprocessor(RequestPreprocessor&&) = delete;
        RequestPreprocessor& operator=(RequestPreprocessor&&) = delete;

    private:
        RawRequestQueue& input_queue_;
        ProcessedSequenceQueue& output_queue_;
        ipc::SharedMemoryManager& shm_manager_;
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

        std::atomic<bool> stop_flag_{false};
    };

} // namespace pie_core::engine
