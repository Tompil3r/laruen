
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>

class BaseArray {
    Shape shape;
    Strides strides;
    uint64_t size;
    uint8_t ndim;
    bool free_mem;
};






#endif