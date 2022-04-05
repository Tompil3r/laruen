
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>


namespace laruen::ndarray::utils {
    inline uint64_t ceil_index(float64_t index) {
        return (uint64_t)index + ((uint64_t)index < index);
    }
}





#endif