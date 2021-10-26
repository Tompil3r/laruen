
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/typenames.h"
#include <cassert>
#include <ostream>
#include <cmath>


template class NDArray<int8_t>;
template class NDArray<uint8_t>;
template class NDArray<int16_t>;
template class NDArray<uint16_t>;
template class NDArray<int32_t>;
template class NDArray<uint32_t>;
template class NDArray<int64_t>;
template class NDArray<uint64_t>;
template class NDArray<float32_t>;
template class NDArray<float64_t>;



template <typename T> NDArray<T>::NDArray()
{
    this->data = nullptr;
    this->shape = {};
    this->strides = {};
    this->ndim = 0;
    this->size = 0;
}


template <typename T> NDArray<T>::NDArray(const Shape &shape)
{
    this->construct_shape(shape);
    this->data = new T[size];
}


template <typename T> NDArray<T>::NDArray(const Shape &shape, T fill_value) : NDArray<T>(shape)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = fill_value;
    }    
}


template <typename T> NDArray<T>::NDArray(T start, T stop, T step) : NDArray<T>({(uint32_t)ceil((stop - start) / step)})
{
    uint64_t idx = 0;

    while(start < stop)
    {
        this->data[idx] = start;
        start += step;
        idx++;
    }
}


template <typename T> T* NDArray<T>::get_data()
{
    return this->data;
}


template <typename T> NDArray<T>::NDArray(T *data, const Shape &shape, const Strides &strides,
uint8_t ndim, uint64_t size)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
    this->ndim = ndim;
    this->size = size;
}