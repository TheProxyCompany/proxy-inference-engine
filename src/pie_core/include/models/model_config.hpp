#pragma once

#include <string>
#include <stdexcept>

namespace pie_core::models {

    struct ModelConfigBase {
        std::string model_type;
    };

    class ConfigParseError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };


    ModelConfigBase parse_model_config_base(const std::string& config_path);

} // namespace pie_core
