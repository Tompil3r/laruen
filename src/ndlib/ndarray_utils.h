
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndlib/ndarray_types.h"
#include <cstdint>


namespace laruen::ndlib::utils {
    inline uint64_t ceil_index(float64_t index) {
        return (uint64_t)index + ((uint64_t)index < index);
    }
}





#endif