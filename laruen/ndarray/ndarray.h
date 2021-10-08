
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/typenames.h"
#include <vector>
#include <cstdint>
#include <ostream>


template <typename T> class NDArray
{
    T *data;
    Shape *shape;
    Strides *strides;
    uint8_t ndim;
    uint64_t size;

        
    public:
        NDArray(const Shape &shape);
        NDArray(const Shape &shape, T fill_value);
        
        const T* getData();

        ~NDArray()
        {
            delete this->data;
            delete this->shape;
            delete this->strides;
        }

        const Shape* getShape() const
        {
            return this->shape;
        }

        const Strides* getStrides() const
        {
            return this->strides;
        }

        uint8_t getNDim() const
        {
            return this->ndim;
        }

        uint64_t getSize() const
        {
            return this->size;
        }
};

#endif