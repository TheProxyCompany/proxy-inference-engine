#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unordered_map.h>
#include <stdexcept>

#include "ipc/request_writer.hpp"
#include "ipc/response_reader.hpp"
#include "ipc/response.hpp"
#include "sequence/sampling_params.hpp"
#include "sequence/logits_params.hpp"
#include "sequence/stop_criteria.hpp"
#include "sequence/ipc_handles.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace pie_core;

// Helper to get the global writer instance, raising Python exception on failure
ipc::RequestWriter* get_writer_instance() {
    try {
        return ipc::get_global_request_writer();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(e.what()); // nanobind converts std::runtime_error
    }
}

// Simplified submit function exposed to Python
uint64_t submit_request_simplified(
    uint64_t request_id,
    const std::string& prompt_string,
    // Sampling Params (basic types)
    float temperature,
    float top_p,
    int top_k,
    float min_p,
    uint32_t rng_seed,
    // Logits Params (basic types/containers)
    float frequency_penalty,
    const std::unordered_map<int32_t, float>& logit_bias,
    float presence_penalty,
    int repetition_context_size,
    float repetition_penalty,
    // Stop Criteria (basic types/containers)
    int max_generated_tokens,
    const std::vector<int32_t>& stop_token_ids,
    // IPC Handles (basic types) - Assuming simple IDs for now
    uint64_t request_channel_id,
    uint64_t response_channel_id,
    // Tool/Format Strings
    const std::string& tool_schemas_str,
    const std::string& response_format_str
) {
    ipc::RequestWriter* writer = get_writer_instance();

    // Construct C++ structs from basic types
    sequence::SamplingParams sampling_params = {
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .rng_seed = rng_seed
    };
    sequence::LogitsParams logits_params = {
        .frequency_penalty = frequency_penalty,
        .logit_bias = logit_bias,
        .presence_penalty = presence_penalty,
        .repetition_context_size = repetition_context_size,
        .repetition_penalty = repetition_penalty
    };
    sequence::StopCriteria stop_criteria = {
        .max_generated_tokens = max_generated_tokens,
        .stop_token_ids = stop_token_ids
    };
     sequence::IPCHandles ipc_handles = {
         .request_channel_id = request_channel_id,
         .response_channel_id = response_channel_id
     };

    // Call the actual C++ writer function
    return writer->submit_request_to_engine(
        request_id,
        prompt_string,
        sampling_params,
        logits_params,
        stop_criteria,
        ipc_handles,
        tool_schemas_str,
        response_format_str
    );
}


NB_MODULE(pie_core, m) {
    m.doc() = "Core C++ module for the Proxy Inference Engine";
    m.def("health_check", []() { return true; });

    // --- Global Writer Management ---
    m.def("init_request_writer", &ipc::init_global_request_writer,
          "Initializes the global RequestWriter instance for submitting requests.");
    m.def("shutdown_request_writer", &ipc::shutdown_global_request_writer,
          "Shuts down and cleans up the global RequestWriter instance.");

    // --- Request Submission ---
    m.def(
        "submit_request", &submit_request_simplified,
        "Submits a request to the inference engine via IPC.",
        "request_id"_a,
        "prompt_string"_a,
        "temperature"_a = 1.0f,
        "top_p"_a = 1.0f, "top_k"_a = -1, "min_p"_a = 0.0f, "rng_seed"_a = 0,
        "frequency_penalty"_a = 0.0f, "logit_bias"_a = std::unordered_map<int32_t, float>{}, "presence_penalty"_a = 0.0f, "repetition_context_size"_a = 60, "repetition_penalty"_a = 1.0f,
        "max_generated_tokens"_a = 1024, "stop_token_ids"_a = std::vector<int32_t>{},
        "request_channel_id"_a = 0, "response_channel_id"_a = 0,
        "tool_schemas_str"_a = "", "response_format_str"_a = ""
    );

    // --- Response Consumption ---
    // Bind the ResponseSlotState enum
     nb::enum_<ipc::ResponseSlotState>(m, "ResponseSlotState")
        .value("FREE_FOR_CPP_WRITER", ipc::ResponseSlotState::FREE_FOR_CPP_WRITER)
        .value("CPP_WRITING", ipc::ResponseSlotState::CPP_WRITING)
        .value("READY_FOR_PYTHON", ipc::ResponseSlotState::READY_FOR_PYTHON)
        .value("PYTHON_READING", ipc::ResponseSlotState::PYTHON_READING)
        .export_values();

     // Bind the FinishReason enum
     nb::enum_<sequence::FinishReason>(m, "FinishReason")
         .value("STOP", sequence::FinishReason::STOP)
         .value("LENGTH", sequence::FinishReason::LENGTH)
         .value("USER", sequence::FinishReason::USER)
         .value("MEMORY", sequence::FinishReason::MEMORY)
         .value("TOOL_USE", sequence::FinishReason::TOOL_USE)
         .value("INJECTION", sequence::FinishReason::INJECTION)
         .export_values();

    // Bind the ResponseReader class - optimized for polling
    nb::class_<ipc::ResponseReader>(m, "ResponseReader")
        .def(
            nb::init<const std::string&>(),
            "response_shm_name"_a = ipc::RESPONSE_QUEUE_SHM_NAME,
            "Initializes the reader to poll the response queue."
        )
        .def(
            "consume_next_delta",
            [](ipc::ResponseReader &reader, int timeout_ms) -> nb::object {
                ipc::ResponseDeltaSlot result_delta{};
                bool success;
                {
                    nb::gil_scoped_release release;
                    success = reader.consume_next_delta(result_delta, timeout_ms);
                }

                if (success) {
                    // Manually create Python dict from C++ data
                    nb::dict py_delta;
                    py_delta["request_id"] = result_delta.request_id;
                    py_delta["num_tokens_in_delta"] = result_delta.num_tokens_in_delta;
                    py_delta["is_final_delta"] = result_delta.is_final_delta;
                    py_delta["finish_reason"] = nb::cast(result_delta.finish_reason);

                    // Convert token array to vector
                    std::vector<int32_t> tokens_vec(
                        result_delta.tokens,
                        result_delta.tokens + result_delta.num_tokens_in_delta
                    );
                    py_delta["tokens"] = nb::cast(tokens_vec);

                    // Convert logprobs array to vector of vectors
                    std::vector<std::vector<float>> logprobs_vec;
                    logprobs_vec.reserve(result_delta.num_tokens_in_delta);
                    for (uint32_t i = 0; i < result_delta.num_tokens_in_delta; ++i) {
                        logprobs_vec.emplace_back(
                            result_delta.logprobs[i],
                            result_delta.logprobs[i] + ipc::MAX_LOGPROBS_PER_TOKEN
                        );
                    }
                    py_delta["logprobs"] = nb::cast(logprobs_vec);
                    return py_delta; // Return dict
                } else {
                    return nb::none(); // Return None on timeout/failure
                }
            },
            "timeout_ms"_a = 0, // Default to non-blocking poll (0ms timeout)
            nb::sig("def consume_next_delta(self, timeout_ms: int = 0) -> dict | None"),
            "Polls for the next available delta. Returns a dict or None on timeout/failure."
        );

    // --- Global Reader Management ---
    m.def("init_response_reader", &ipc::init_global_response_reader,
          "Initializes the global ResponseReader instance.",
          "response_shm_name"_a = ipc::RESPONSE_QUEUE_SHM_NAME,
          nb::sig("def init_response_reader(response_shm_name: str = ...) -> None"));
    m.def("shutdown_response_reader", &ipc::shutdown_global_response_reader,
          "Shuts down and cleans up the global ResponseReader instance.",
          nb::sig("def shutdown_response_reader() -> None"));
}
