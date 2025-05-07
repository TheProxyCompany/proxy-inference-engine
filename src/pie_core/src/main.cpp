#include <iostream>
#include <string>
#include <spdlog/spdlog.h>
#include "engine/engine.hpp"

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

    try {
        pie_core::engine::Engine engine(model_path);
        engine.start();
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }
}
