#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <tokenizers_cpp.h>

namespace pie_core::tokenizers {

    // Create a clear alias for the HuggingFace tokenizer implementation
    using HuggingFaceTokenizer = ::tokenizers::Tokenizer;

    /**
     * @brief Exception thrown for tokenizer-related errors
     */
    class TokenizerException : public std::runtime_error {
    public:
        explicit TokenizerException(const std::string& message) : std::runtime_error(message) {}
    };

    /**
     * @brief Wrapper around tokenizers-cpp interface, providing a unified tokenization interface
     */
    class Tokenizer {
    public:
        /**
         * @brief Constructor that initializes a tokenizer from a model path
         * @param model_path Path to directory containing tokenizer.json or tokenizer.model
         * @throws TokenizerException if initialization fails
         */
        explicit Tokenizer(const std::string& model_path);

        /**
         * @brief Tokenizes input text into token IDs
         * @param text The input text to encode
         * @return Vector of token IDs
         * @throws TokenizerException on tokenization failure
         */
        std::vector<int32_t> encode(const std::string& text) const;

        /**
         * @brief Detokenizes token IDs back into text
         * @param ids The token IDs to decode
         * @return Decoded text
         * @throws TokenizerException on detokenization failure
         */
        std::string decode(const std::vector<int32_t>& ids) const;

        /**
         * @brief Access the underlying tokenizer object
         * @return Pointer to the internal tokenizer instance
         */
        const HuggingFaceTokenizer* get_internal_tokenizer() const;

        static std::unique_ptr<Tokenizer> FromBlobJSON(const std::vector<uint8_t>& blob);
        static std::unique_ptr<Tokenizer> FromBlobModel(const std::vector<uint8_t>& blob);
        static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::vector<uint8_t>& blob);

        // Prevent copying and moving
        Tokenizer(const Tokenizer&) = delete;
        Tokenizer& operator=(const Tokenizer&) = delete;
        Tokenizer(Tokenizer&&) = delete;
        Tokenizer& operator=(Tokenizer&&) = delete;

    private:
        std::unique_ptr<HuggingFaceTokenizer> tokenizer_;
    };

} // namespace pie_core::tokenizers
