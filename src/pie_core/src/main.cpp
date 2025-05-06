#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include "models/model_factory.hpp"
#include "ipc/ipc_producer.hpp"
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
        if (!ipc_name.empty()) {
            spdlog::info("IPC mode enabled with shared memory name: {}", ipc_name);
            std::unique_ptr<pie_core::ipc::SharedMemoryManager> bulk_data_shm_manager;
            try {
                bulk_data_shm_manager = std::make_unique<pie_core::ipc::SharedMemoryManager>(
                    pie_core::ipc::BULK_DATA_SHM_NAME,
                    pie_core::ipc::BULK_DATA_SHM_SIZE,
                    true
                );
            } catch (const pie_core::ipc::SharedMemoryError& e) {
                std::cerr << "FATAL: Failed to initialize bulk data SHM: " << e.what() << std::endl;
                return 1;
            }
        }
        spdlog::info("Engine initialized successfully");
        // Main loop would go here
        spdlog::info("Engine shutting down");
        return 0;
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }
}
