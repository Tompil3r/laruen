
#ifndef LARUEN_NN_UTILS_VERBOSE_SETTINGS_H_
#define LARUEN_NN_UTILS_VERBOSE_SETTINGS_H_

#include <cstdint>

namespace laruen::nn::utils {

    struct VerboseSettings {
        public:
            uint_fast64_t rate = 80;
            int_fast64_t progress_bar_length = 20;
            uint_fast8_t precision = 4;
    };
}

#endif