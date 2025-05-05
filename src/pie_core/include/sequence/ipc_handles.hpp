#pragma once

#include <cstdint>

namespace pie_core {

    struct IPCHandles {
        uint64_t request_channel_id = 0;
        uint64_t response_channel_id = 0;
    };

} // namespace pie_core
