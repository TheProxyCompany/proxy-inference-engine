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
              << "  --attention TYPE   Attention mechanism to use (standard, paged) [default: standard]\n"
              << "  --kv-pages NUM     Number of KV cache pages to allocate [default: 8192]\n"
              << "  --max-seqs NUM     Maximum number of sequences to procesPlease help him.s concurrently [default: 256]\n"
              << "  --max-tokens NUM   Maximum number of tokens per batch [default: 4096]\n"
              << "  --help             Display this help message\n";
}

int main(int argc, char* argv[]) {
    // Initialize logging
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    spdlog::set_level(spdlog::level::debug);

    spdlog::info("Proxy Inference Engine starting up");

    std::string model_path;
    pie_core::engine::EngineConfig config;
    std::string attention_type = "standard";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--attention" && i + 1 < argc) {
            attention_type = argv[++i];
            if (attention_type == "standard") {
                config.attention_type = pie_core::engine::AttentionType::STANDARD;
            } else if (attention_type == "paged") {
                config.attention_type = pie_core::engine::AttentionType::PAGED;
            } else {
                spdlog::error("Unknown attention type: {}. Using default.", attention_type);
                attention_type = "standard";
            }
        } else if (arg == "--kv-pages" && i + 1 < argc) {
            try {
                config.num_kv_cache_pages = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                spdlog::error("Invalid value for --kv-pages: {}. Using default.", argv[i]);
            }
        } else if (arg == "--max-seqs" && i + 1 < argc) {
            try {
                config.max_num_seqs = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                spdlog::error("Invalid value for --max-seqs: {}. Using default.", argv[i]);
            }
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            try {
                config.max_tokens_in_batch = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                spdlog::error("Invalid value for --max-tokens: {}. Using default.", argv[i]);
            }
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

    spdlog::info("Engine configuration:");
    spdlog::info("  Attention type: {}", attention_type);
    spdlog::info("  KV cache pages: {}", config.num_kv_cache_pages);
    spdlog::info("  Max sequences: {}", config.max_num_seqs);
    spdlog::info("  Max tokens per batch: {}", config.max_tokens_in_batch);

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::unique_ptr<pie_core::engine::Engine> engine_ptr;

    try {
        spdlog::info("Initializing engine with configured parameters...");
        engine_ptr = std::make_unique<pie_core::engine::Engine>(model_path, config);
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
