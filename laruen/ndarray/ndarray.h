
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/dtype.h"
#include <vector>
#include <cassert>
#include <cstdint>

#define MAX_NDIM 32


template<class T>
class NDArray
{
    T *data;
    std::vector<uint32_t> *shape;
    std::vector<uint64_t> *strides;
    uint8_t ndim;
    uint64_t size;
    DType::DType &dtype;
};



#endif