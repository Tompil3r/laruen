
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/typenames.h"
#include <cassert>


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


template <typename T> NDArray<T>::NDArray(const Shape &shape)
{
    uint8_t ndim = shape.size();
    uint64_t stride = 1;
    uint64_t size = shape[ndim - 1];
    Strides *strides = new Strides(ndim);
    (*strides)[ndim - 1] = stride;
    
    for(int idx = ndim - 1;idx-- > 0;)
    {
        stride *= shape[idx];
        (*strides)[idx] = stride;
        size *= shape[idx];
    }

    this->data = new T[size];
    this->ndim = ndim;
    this->shape = new Shape(shape);
    this->size = size;
    this->strides = strides;
}