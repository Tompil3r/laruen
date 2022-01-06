
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "laruen/ndarray/ndarray_types.h"
#include <cstdint>


namespace laruen::ndarray::utils
{
    inline uint32_t ceil_index(float64_t index)
    {
        return (uint32_t)index + ((uint32_t)index < index);
    }
}





#endif