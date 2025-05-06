#pragma once

#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp" // For response path info

#include <string>
#include <vector>
#include <cstdint>

namespace pie_core::engine {

    enum class PromptType {
        SINGLE_STRING,
        CHAT_HISTORY,
    };

    struct RawRequestData {
        uint64_t request_id;
        std::string prompt_payload; // Raw prompt string or structured chat string (e.g., JSON)
        PromptType type;            // To guide the preprocessor

        // Parameters directly from the RequestSlot or processed by IPCReader
        sequence::SamplingParams sampling_params;
        sequence::LogitsParams logits_params;
        sequence::StopCriteria stop_criteria;
        sequence::IPCHandles ipc_handles; // For routing the response

        std::string tool_schemas_json_str; // If tools are passed as strings
        std::string response_format_json_str;

        // Timestamp from IPCReader for arrival at C++ engine
        uint64_t arrival_timestamp_ns;
    };

} // namespace pie_core::engine
