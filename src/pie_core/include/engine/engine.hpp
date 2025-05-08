#pragma once
#include "models/imodel.hpp"
#include "engine/request_preprocessor.hpp"
#include "engine/response_postprocessor.hpp"
#include "engine/page_allocator.hpp"
#include "engine/scheduler.hpp"
#include "engine/batch_details.hpp"
#include "ipc/request_reader.hpp"
#include "ipc/response_writer.hpp"
#include "ipc/request_writer.hpp"
#include "ipc/shared_memory_manager.hpp"
#include "ipc/ipc_manager.hpp"
#include "tokenizers/tokenizer.hpp"
#include <atomic>

namespace pie_core::engine {

    /**
     * @brief Configuration struct for Engine parameters
     */
    struct EngineConfig {
        // KV Cache configuration
        size_t num_kv_cache_pages = 8192;

        // Scheduler configuration
        size_t max_num_seqs = 256;
        size_t max_tokens_in_batch = 4096;

        // Attention mechanism configuration
        AttentionType attention_type = AttentionType::STANDARD;
    };

    class Engine {
    public:
        /**
         * @brief Constructs the Engine with default configuration
         * @param model_path Path to the model directory
         */
        Engine(const std::string& model_path);

        /**
         * @brief Constructs the Engine with custom configuration
         * @param model_path Path to the model directory
         * @param config Custom engine configuration
         */
        Engine(const std::string& model_path, const EngineConfig& config);
        ~Engine();

        /**
         * @brief Runs the engine in a blocking manner until shutdown is signaled
         */
        void run_blocking();

        /**
         * @brief Signals all components to stop and joins component threads
         */
        void stop();

    private:
        std::thread reader_t_;
        std::thread preprocessor_t_;
        std::thread scheduler_t_;
        std::thread postprocessor_t_;

        std::atomic<bool> stop_flag_{false};
        EngineConfig config_; // Engine configuration

        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
        std::unique_ptr<models::IModel> model_;
        std::unique_ptr<ipc::SharedMemoryManager> bulk_shm_manager_;
        std::unique_ptr<ipc::RequestReader> request_reader_;
        std::unique_ptr<ipc::ResponseWriter> response_writer_;
        std::unique_ptr<ipc::IPCManager> ipc_manager_;
        std::unique_ptr<PageAllocator> allocator_;
        std::unique_ptr<RequestPreprocessor> preprocessor_;
        std::unique_ptr<ResponsePostprocessor> postprocessor_;
        std::unique_ptr<Scheduler> scheduler_;

        // Queues for inter-thread communication
        ipc::RequestReader::RawRequestQueue raw_request_queue_;
        engine::RequestPreprocessor::ProcessedSequenceQueue processed_sequence_queue_;
        engine::PostprocessingQueue postprocessing_queue_;
    };

} // namespace pie_core::engine
