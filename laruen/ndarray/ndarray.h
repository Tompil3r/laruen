
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
        
        const T* get_data();

        ~NDArray()
        {
            delete this->data;
            delete this->shape;
            delete this->strides;
        }

        const Shape* get_shape() const
        {
            return this->shape;
        }

        const Strides* get_strides() const
        {
            return this->strides;
        }

        uint8_t get_ndim() const
        {
            return this->ndim;
        }

        uint64_t get_size() const
        {
            return this->size;
        }
};

#endif