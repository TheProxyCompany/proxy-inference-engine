#pragma once

#include "models/imodel.hpp"
#include <string>
#include <memory>
#include <stdexcept>

namespace pie_core {

    class ModelLoadError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    std::unique_ptr<IModel> load_model(const std::string& model_path);

} // namespace pie_core
