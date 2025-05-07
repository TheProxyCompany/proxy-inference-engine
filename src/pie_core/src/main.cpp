#include <iostream>
#include <string>
#include <memory>
#include <csignal>
#include <atomic>
#include <spdlog/spdlog.h>
#include "engine/engine.hpp"

// Global atomic flag to signal shutdown
std::atomic<bool> g_shutdown_requested = false;

// Signal handler function
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // This is the ONLY safe thing to do in the handler
        g_shutdown_requested.store(true, std::memory_order_relaxed);
        // Do NOT log or call complex functions here!
    }
}

void print_usage() {
    std::cout << "Usage: pie_engine [options]\n"
              << "Options:\n"
              << "  --model PATH       Path to model directory or Hugging Face repository ID\n"
              << "  --help             Display this help message\n";
}

int main(int argc, char* argv[]) {
    // Initialize logging
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    spdlog::set_level(spdlog::level::info);

    spdlog::info("Proxy Inference Engine starting up");

    std::string model_path;
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
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

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::unique_ptr<pie_core::engine::Engine> engine_ptr;

    try {
        spdlog::info("Initializing engine...");
        engine_ptr = std::make_unique<pie_core::engine::Engine>(model_path);
        spdlog::info("Engine initialization complete. Starting run loop...");

        // Run the engine's blocking loop, which will internally check the flag
        engine_ptr->run_blocking();

        spdlog::info("Engine run loop exited normally.");
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error during engine initialization or run: {}", e.what());
        // engine_ptr destructor will handle cleanup if partially initialized
        return 1;
    }

    spdlog::info("Initiating final cleanup...");
    // Destructor of engine_ptr is called automatically when main exits,
    // which will call engine_ptr->stop() if not already stopped
}
