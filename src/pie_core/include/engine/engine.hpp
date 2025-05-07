#pragma once
#include "models/imodel.hpp"
#include "engine/request_preprocessor.hpp"
#include "engine/page_allocator.hpp"
#include "engine/scheduler.hpp"
#include "ipc/request_reader.hpp"
#include "ipc/response_writer.hpp"
#include "ipc/request_writer.hpp"
#include "ipc/shared_memory_manager.hpp"
#include "ipc/ipc_manager.hpp"
#include <atomic>

namespace pie_core::engine {

    class Engine {
    public:
        Engine(const std::string& model_path);
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
        std::thread writer_t_;

        std::atomic<bool> stop_flag_{false};

        std::unique_ptr<models::IModel> model_;
        std::unique_ptr<ipc::SharedMemoryManager> bulk_shm_manager_;
        std::unique_ptr<ipc::RequestReader> request_reader_;
        std::unique_ptr<ipc::ResponseWriter> response_writer_;
        std::unique_ptr<ipc::IPCManager> ipc_manager_;
        std::unique_ptr<PageAllocator> allocator_;
        std::unique_ptr<RequestPreprocessor> preprocessor_;
        std::unique_ptr<Scheduler> scheduler_;
        ipc::RequestReader::RawRequestQueue raw_request_queue_;
        engine::RequestPreprocessor::ProcessedSequenceQueue processed_sequence_queue_;
    };

} // namespace pie_core::engine
