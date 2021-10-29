
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/typenames.h"
#include <cassert>
#include <ostream>
#include <cmath>

using namespace laruen::ndarray;

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
    this->delete_data = true;
}

template <typename T> NDArray<T>::NDArray(const Shape &shape)
{
    this->construct_shape(shape);
    this->data = new T[size];
    this->delete_data = true;
}

template <typename T> NDArray<T>::NDArray(const Shape &shape, T fill_value) : NDArray<T>(shape)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = fill_value;
    }    
}

template <typename T> NDArray<T>::NDArray(T *data, const Shape &shape, const Strides &strides,
uint8_t ndim, uint64_t size, bool delete_data)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
    this->ndim = ndim;
    this->size = size;
    this->delete_data = delete_data;
}

template <typename T> NDArray<T>::NDArray(const NDArray<T> &ndarray) : NDArray<T>(nullptr,
ndarray.get_shape(), ndarray.get_strides(), ndarray.get_ndim(), ndarray.get_size(), true)
{
    const T *data = ndarray.get_data();
    T *copied_data = new T[this->size];

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        copied_data[idx] = data[idx];
    }

    this->data = copied_data;
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

template <typename T> const T* NDArray<T>::get_data() const
{
    return this->data;
}

template <typename T> uint64_t NDArray<T>::ravel_ndindex(const NDIndex& ndindex)
{
    uint64_t index = 0;
    uint8_t nb_dims = ndindex.size();

    for(uint8_t dim = 0;dim < nb_dims;dim++)
    {
        index += ndindex[dim] * this->strides[dim];
    }

    return index;
}

template <typename T> NDIndex NDArray<T>::unravel_index(uint64_t index)
{
    NDIndex ndindex;
    ndindex.reserve(this->ndim);

    for(uint8_t dim = 0;dim < this->ndim;dim++)
    {
        ndindex.push_back(index / this->strides[dim]);
        index -= ndindex[dim] * this->strides[dim];
    }

    return ndindex;
}


template <typename T> T& NDArray<T>::operator[](uint64_t index)
{
    return this->data[index];
}

template <typename T> const T& NDArray<T>::operator[](uint64_t index) const
{
    return this->data[index];
}

template <typename T> T& NDArray<T>::operator[](const NDIndex &ndindex)
{
    uint64_t data_index = 0;
    uint64_t nb_dims = ndindex.size();

    for(uint32_t dim = 0;dim < nb_dims;dim++)
    {
        data_index += ndindex[dim] * this->strides[dim];
    }

    return this->data[data_index];
}

template <typename T> const T& NDArray<T>::operator[](const NDIndex &ndindex) const
{
    uint64_t data_index = 0;
    uint64_t nb_dims = ndindex.size();

    for(uint32_t dim = 0;dim < nb_dims;dim++)
    {
        data_index += ndindex[dim] * this->strides[dim];
    }

    return this->data[data_index];
}