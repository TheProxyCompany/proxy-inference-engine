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
#include <atomic>
#include <chrono>
#include <thread>

// Declaration for the global shutdown flag defined in main.cpp
extern std::atomic<bool> g_shutdown_requested;

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

        // --- 7. Initialize Response Writer ---
        spdlog::info("Engine: Initializing Response Writer...");
        response_writer_ = std::make_unique<ipc::ResponseWriter>(ipc::RESPONSE_QUEUE_SHM_NAME);
        spdlog::info("Engine: Response Writer initialized.");

        // --- 8. Initialize Scheduler ---
        spdlog::info("Engine: Initializing Scheduler...");
        scheduler_ = std::make_unique<engine::Scheduler>(
            *allocator_,
            *model_, // Pass by reference, not move the unique_ptr
            processed_sequence_queue_,
            *response_writer_,
            /* max_num_seqs= */ 256,
            /* max_tokens_in_batch= */ 4096
        );
        spdlog::info("Engine: Scheduler initialized.");

        // Done initializing all components
        spdlog::info("Engine: Full initialization complete.");
    }


    Engine::~Engine() {
        spdlog::info("Engine: Destructor called, ensuring shutdown...");
        stop(); // Ensure stop sequence is called if not already
        spdlog::info("Engine: Shutdown complete.");
    }

    void Engine::run_blocking() {
        spdlog::info("Engine: Starting component threads...");
        if (!request_reader_ || !preprocessor_ || !scheduler_) {
            throw std::runtime_error("Engine cannot run: Components not initialized.");
        }

        // Start component threads, calling their run_loop methods
        reader_t_ = std::thread([this] { request_reader_->run_loop(); });
        preprocessor_t_ = std::thread([this] { preprocessor_->run_loop(); });
        scheduler_t_ = std::thread([this] { scheduler_->run_loop(); });
        spdlog::info("Engine: Component threads started.");

        spdlog::info("Engine: Running... (Waiting for shutdown signal via atomic flag)");
        // Loop until the global shutdown flag is set by the signal handler
        while (!g_shutdown_requested.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Check periodically
        }
        spdlog::info("Engine: Shutdown signal detected via atomic flag. Initiating stop sequence.");

        // Trigger the stop sequence
        this->stop();
    }

    void Engine::stop() {
        // Use exchange to ensure stop logic runs only once
        if (stop_flag_.exchange(true, std::memory_order_acq_rel)) {
            spdlog::debug("Engine: Stop already initiated.");
            return; // Already stopping/stopped
        }
        spdlog::info("Engine: Signaling components to stop...");

        // Signal component loops to stop first
        if (request_reader_) request_reader_->stop();
        if (preprocessor_) preprocessor_->stop();
        if (scheduler_) scheduler_->stop();

        // Wake up the reader thread if it's blocked in kevent/eventfd_read
        if (ipc_manager_) ipc_manager_->trigger_kernel_event();

        spdlog::info("Engine: Joining component threads...");
        // Join threads
        if (scheduler_t_.joinable()) {
            scheduler_t_.join();
            spdlog::debug("Engine: Scheduler thread joined.");
        }
        if (preprocessor_t_.joinable()) {
            preprocessor_t_.join();
            spdlog::debug("Engine: Preprocessor thread joined.");
        }
        if (reader_t_.joinable()) {
            reader_t_.join();
            spdlog::debug("Engine: Reader thread joined.");
        }
        spdlog::info("Engine: All component threads joined.");
    }


} // namespace pie_core::engine
