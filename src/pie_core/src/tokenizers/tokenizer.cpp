#include "tokenizers/tokenizer.hpp"
#include "utils/read_file.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>

namespace pie_core::tokenizers {

namespace fs = std::filesystem;

Tokenizer::Tokenizer(const std::string& model_path) {
    spdlog::info("Tokenizer: Initializing with model_path='{}'", model_path);

    fs::path model_directory = fs::path(model_path);
    if (!fs::exists(model_directory)) {
        spdlog::critical("Tokenizer: Model directory does not exist: {}", model_directory.string());
        throw TokenizerException("Model directory does not exist: " + model_directory.string());
    }
    spdlog::debug("Tokenizer: Model directory exists at: {}", model_directory.string());

    // Load tokenizer from model directory
    spdlog::info("Tokenizer: Attempting to load tokenizer from model directory");

    if (fs::exists(model_directory / "tokenizer.json")) {
        const auto tokenizer_file_path = (model_directory / "tokenizer.json").string();
        spdlog::debug("Tokenizer: Found tokenizer.json at: {}", tokenizer_file_path);

        try {
            auto file_contents_blob = load_file_bytes(tokenizer_file_path);
            spdlog::debug("Tokenizer: Loaded tokenizer.json file, size: {} bytes", file_contents_blob.size());

            tokenizer_ = std::unique_ptr<HuggingFaceTokenizer>(
                HuggingFaceTokenizer::FromBlobJSON(file_contents_blob)
            );
            spdlog::info("Tokenizer: Successfully initialized JSON tokenizer from '{}'", tokenizer_file_path);
        } catch (const std::exception& e) {
            spdlog::critical("Tokenizer: Failed to load JSON tokenizer from '{}': {}",
                tokenizer_file_path, e.what());
            throw TokenizerException("Failed to load JSON tokenizer: " + std::string(e.what()));
        }
    } else if (fs::exists(model_directory / "tokenizer.model")) {
        const auto tokenizer_file_path = (model_directory / "tokenizer.model").string();
        spdlog::debug("Tokenizer: Found tokenizer.model at: {}", tokenizer_file_path);

        try {
            auto file_contents_blob = load_file_bytes(tokenizer_file_path);
            spdlog::debug("Tokenizer: Loaded tokenizer.model file, size: {} bytes", file_contents_blob.size());

            tokenizer_ = std::unique_ptr<HuggingFaceTokenizer>(
                HuggingFaceTokenizer::FromBlobSentencePiece(file_contents_blob)
            );
            spdlog::info("Tokenizer: Successfully initialized SentencePiece tokenizer from '{}'", tokenizer_file_path);
        } catch (const std::exception& e) {
            spdlog::critical("Tokenizer: Failed to load SentencePiece tokenizer from '{}': {}",
                tokenizer_file_path, e.what());
            throw TokenizerException("Failed to load SentencePiece tokenizer: " + std::string(e.what()));
        }
    } else {
        spdlog::critical("Tokenizer: No tokenizer.json or tokenizer.model found in {}", model_directory.string());
        throw TokenizerException("No tokenizer.json or tokenizer.model found in " + model_directory.string());
    }

    spdlog::info("Tokenizer: Initialization complete");
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (!tokenizer_) {
        throw TokenizerException("Tokenizer not initialized");
    }

    try {
        spdlog::trace("Tokenizer: Encoding text of size {} bytes", text.size());
        auto token_ids = tokenizer_->Encode(text);

        if (token_ids.empty() && !text.empty()) {
            spdlog::warn("Tokenizer: Encoding produced empty token list for non-empty text");
        } else {
            spdlog::trace("Tokenizer: Successfully encoded text to {} tokens", token_ids.size());
        }

        return token_ids;
    } catch (const std::exception& e) {
        spdlog::error("Tokenizer: Failed to encode text: {}", e.what());
        throw TokenizerException("Failed to encode text: " + std::string(e.what()));
    }
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    if (!tokenizer_) {
        throw TokenizerException("Tokenizer not initialized");
    }

    try {
        spdlog::trace("Tokenizer: Decoding {} tokens", ids.size());
        auto text = tokenizer_->Decode(ids);
        spdlog::trace("Tokenizer: Successfully decoded {} tokens to {} bytes",
            ids.size(), text.size());
        return text;
    } catch (const std::exception& e) {
        spdlog::error("Tokenizer: Failed to decode tokens: {}", e.what());
        throw TokenizerException("Failed to decode tokens: " + std::string(e.what()));
    }
}

const HuggingFaceTokenizer* Tokenizer::get_internal_tokenizer() const {
    return tokenizer_.get();
}

} // namespace pie_core::tokenizers
