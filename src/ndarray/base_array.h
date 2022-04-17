
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>

class BaseArray {
    Shape m_shape;
    Strides m_strides;
    uint64_t m_size;
    uint8_t m_ndim;
    bool m_free_mem;
};






#endif