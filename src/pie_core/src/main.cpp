#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include "models/model_factory.hpp"
#include "engine/page_allocator.hpp"
#include "engine/request_preprocessor.hpp"
#include "ipc/request_writer.hpp"
#include "ipc/request_reader.hpp"
#include "ipc/shared_memory_manager.hpp"

using json = nlohmann::json;

void print_usage() {
    std::cout << "Usage: pie_engine [options]\n"
              << "Options:\n"
              << "  --model PATH       Path to model or Hugging Face repository ID\n"
              << "  --ipc NAME         Enable IPC mode with shared memory segment name\n"
              << "  --help             Display this help message\n";
}

int main(int argc, char* argv[]) {
    // Initialize logging
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    spdlog::set_level(spdlog::level::info);

    spdlog::info("Proxy Inference Engine starting up");

    std::string model_path;
    std::string ipc_name;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--ipc" && i + 1 < argc) {
            ipc_name = argv[++i];
        } else {
            spdlog::error("Unknown argument: {}", arg);
            print_usage();
            return 1;
        }
    }

    if (model_path.empty()) {
        spdlog::error("No model path specified");
        print_usage();
        return 1;
    }

    try {
        spdlog::info("Loading model from: {}", model_path);
        // ---------- heavyweight objects ----------
        auto model = pie_core::models::load_model(model_path);
        pie_core::engine::PageAllocator allocator(
            8192,
            model->get_num_kv_heads(),
            model->get_head_dim()
        );

        // ---------- IPC mode ----------
        auto bulk_data_shm_manager = std::make_unique<pie_core::ipc::SharedMemoryManager>(
            pie_core::ipc::BULK_DATA_SHM_NAME,
            pie_core::ipc::BULK_DATA_SHM_SIZE,
            true
        );

        using RawQ = pie_core::ipc::RequestReader::RawRequestQueue;
        using SeqQ = pie_core::engine::RequestPreprocessor::ProcessedSequenceQueue;
        RawQ raw_q;   SeqQ seq_q;

        pie_core::ipc::RequestReader reader(raw_q, *bulk_data_shm_manager, ipc_name);
        std::thread reader_t([&]{ reader.run(); });

        pie_core::engine::RequestPreprocessor preproc(raw_q, seq_q, *bulk_data_shm_manager, model_path);
        preproc.start();

        spdlog::info("Engine initialized successfully");
        // Main loop would go here
        spdlog::info("Engine shutting down");
        // ---------- shutdown ----------
        reader.stop();
        reader_t.join();
        preproc.stop_and_join();
        spdlog::info("Bye!");
        return 0;
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }
}
