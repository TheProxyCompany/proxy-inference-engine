#include "models/model_config.hpp"
#include <nlohmann/json.hpp>

namespace pie_core::models {

    ModelConfigBase parse_model_config_base(const std::string& config_path) {
        return ModelConfigBase();
    }

} // namespace pie_core::models
