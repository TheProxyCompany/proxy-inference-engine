#include "engine/engine.hpp"
#include "ipc/request_reader.hpp"
#include "ipc/response_writer.hpp"
#include "ipc/request.hpp"
#include "ipc/response.hpp"
#include "ipc/ipc_manager.hpp"
#include "engine/page_allocator.hpp"
#include "engine/request_preprocessor.hpp"
#include "engine/scheduler.hpp"
#include "models/model_factory.hpp"
#include "spdlog/spdlog.h"

namespace pie_core::engine {

    Engine::Engine(const std::string& model_path)
    {
        // --- 1. Initialize Inter-Process Communication (IPC) ---
        try {
            ipc_manager_ = std::make_unique<ipc::IPCManager>(
                ipc::REQUEST_QUEUE_SHM_NAME,
                ipc::RESPONSE_QUEUE_SHM_NAME
            );
            spdlog::info("Engine: IPC Manager initialized.");
        } catch (const ipc::IPCError& e) {
            spdlog::critical("Engine: Failed to initialize IPC Manager: {}", e.what());
            throw; // Re-throw critical error
        }

        // --- 2. Initialize bulk data shared memory manager ---
        spdlog::info("Initializing bulk data shared memory manager with name: {}", ipc::BULK_DATA_SHM_NAME);
        bulk_shm_manager_ = std::make_unique<ipc::SharedMemoryManager>(
            ipc::BULK_DATA_SHM_NAME,
            ipc::BULK_DATA_SHM_SIZE,
            true
        );
        spdlog::info("Bulk data shared memory manager initialized");

        // --- 3. Load Model ---
        spdlog::info("Loading model from: {}", model_path);
        model_ = models::load_model(model_path);
        spdlog::info("Model loaded with {} layers, {} kv heads, {} head dim, {} vocab size",
            model_->get_num_layers(),
            model_->get_num_kv_heads(),
            model_->get_head_dim(),
            model_->get_vocab_size()
        );

        // --- 4. Initialize KV Cache Page Allocator ---
        spdlog::info("Initializing paged KV cache allocator with {} pages", 8192);
        allocator_ = std::make_unique<engine::PageAllocator>(
            8192,
            model_->get_num_kv_heads(),
            model_->get_head_dim()
        );
        spdlog::info("KV cache allocator initialized");

        // --- 5. Initialize Request Reader ---
        spdlog::info("Engine: Initializing Request Reader...");
        request_reader_ = std::make_unique<ipc::RequestReader>(
            raw_request_queue_,
            *bulk_shm_manager_,
            ipc::REQUEST_QUEUE_SHM_NAME,
            ipc_manager_->get_kernel_event_fd()
        );
        spdlog::info("Engine: Request Reader initialized.");

        // --- 6. Initialize Request Preprocessor ---
        spdlog::info("Engine: Initializing Request Preprocessor...");
        preprocessor_ = std::make_unique<engine::RequestPreprocessor>(
            raw_request_queue_,
            processed_sequence_queue_,
            *bulk_shm_manager_,
            model_path
        );
        spdlog::info("Engine: Request Preprocessor initialized.");

        // --- 7. Initialize Scheduler ---
        spdlog::info("Engine: Initializing Scheduler...");
        scheduler_ = std::make_unique<engine::Scheduler>(
            *allocator_,
            std::move(model_),
            /* max_num_seqs= */ 256,
            /* max_tokens_in_batch= */ 4096
        );
        spdlog::info("Engine: Scheduler initialized.");


        // --- 8. Initialize Response Writer ---
        spdlog::info("Engine: Initializing Response Writer...");
        // TODO: Implement ResponseWriter class
        // response_writer_ = std::make_unique<ipc::ResponseWriter>(/* args */);
        // spdlog::info("Engine: Response Writer initialized.");
        spdlog::info("Engine: Response Writer initialized.");

        // Done initializing all components
        spdlog::info("Engine: Full initialization complete.");
    }


    Engine::~Engine() {
        spdlog::info("Engine: Shutting down...");
        // Stop components in reverse order of dependency/start
        // if (scheduler_) scheduler_->stop();
        if (preprocessor_) preprocessor_->stop_and_join();
        if (request_reader_) request_reader_->stop();
        if (reader_t_.joinable()) reader_t_.join();
        if (preprocessor_t_.joinable()) preprocessor_t_.join();
        // Other cleanup if needed
        spdlog::info("Engine: Shutdown complete.");
    }

    void Engine::start() {
        spdlog::info("Engine: Starting components...");
        // Start threads/loops if not started in constructors
        reader_t_ = std::thread([&] { request_reader_->start(); });
        preprocessor_t_ = std::thread([&] { preprocessor_->start(); });
        // scheduler_t_ = std::thread([&] { scheduler_->start(); });
        // writer_t_ = std::thread([&] { response_writer_->start(); });
        spdlog::info("Engine: Components started.");
    }


} // namespace pie_core::engine
