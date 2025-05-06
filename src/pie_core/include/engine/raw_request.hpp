#pragma once

#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"

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
        std::string prompt_payload;
        uint64_t _shm_prompt_offset;
        uint64_t _shm_prompt_size;
        PromptType type;

        sequence::SamplingParams sampling_params;
        sequence::LogitsParams logits_params;
        sequence::StopCriteria stop_criteria;
        sequence::IPCHandles ipc_handles;

        std::string tool_schemas_str;
        std::string response_format_str;

        uint64_t arrival_timestamp_ns;
    };

} // namespace pie_core::engine
