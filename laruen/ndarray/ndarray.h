
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/typenames.h"
#include <vector>
#include <cstdint>


template <typename T>
class NDArray
{
    T *data;
    Shape *shape;
    Strides *strides;
    uint8_t ndim;
    uint64_t size;

        
    public:
        NDArray(const Shape &shape);
};


#endif