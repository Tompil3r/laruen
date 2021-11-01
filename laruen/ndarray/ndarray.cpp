
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/typenames.h"
#include "laruen/ndarray/ndarray_utils.h"
#include "laruen/utils/range.h"
#include <cassert>
#include <ostream>
#include <cmath>

using namespace laruen::ndarray;
using namespace laruen::ndarray::utils;

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


template <typename T> NDArray<T>::~NDArray()
{
    if(this->delete_data)
    {
        delete this->data;
    }
}

template <typename T> NDArray<T>::NDArray()
{
    this->data = nullptr;
    this->ndim = 0;
    this->size = 0;
    this->delete_data = true;
}

template <typename T> NDArray<T>::NDArray(const Shape &shape)
{
    this->shape_array(shape);
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

template <typename T> const Shape& NDArray<T>::get_shape() const
{
    return this->shape;
}

template <typename T> const Strides& NDArray<T>::get_strides() const
{
    return this->strides;
}

template <typename T> uint8_t NDArray<T>::get_ndim() const
{
    return this->ndim;
}

template <typename T> uint64_t NDArray<T>::get_size() const
{
    return this->size;
}

template <typename T> bool NDArray<T>::does_delete_data()
{
    return this->delete_data;
}

template <typename T> void NDArray<T>::set_delete_data(bool delete_date)
{
    this->delete_data = delete_data;
}

template <typename T> NDArray<T> NDArray<T>::shallow_copy()
{
    return NDArray<T>(this->data, this->shape, this->strides, this->ndim, this->size, false);
}

template <typename T> const NDArray<T> NDArray<T>::shallow_copy() const
{
    return NDArray<T>(this->data, this->shape, this->strides, this->ndim, this->size, false);
}

template <typename T> void NDArray<T>::reshape(const Shape &shape)
{
    uint64_t stride = this->strides[this->ndim - 1];
    this->ndim = shape.size();
    uint64_t size = shape[this->ndim - 1];

    this->strides = Strides(this->ndim);
    this->strides[this->ndim - 1] = stride;

    for(uint8_t dim = this->ndim - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        this->strides[dim] = stride;
        size *= shape[dim];
    }

    assert(this->size == size);
    this->shape = Shape(shape);
}

template <typename T> uint64_t NDArray<T>::ravel_ndindex(const NDIndex &ndindex) const
{
    uint64_t index = 0;
    uint8_t nb_dims = ndindex.size();

    for(uint8_t dim = 0;dim < nb_dims;dim++)
    {
        index += ndindex[dim] * this->strides[dim];
    }

    return index;
}

template <typename T> NDIndex NDArray<T>::unravel_index(uint64_t index) const
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

template <typename T> std::string NDArray<T>::get_specs() const
{
    std::ostringstream specs;
    uint8_t dim;

    specs << "shape=(";
    for(dim = 0;dim < this->ndim - 1;dim++) specs << this->shape[dim] << ',' << ' ';
    specs << this->shape[dim] << ")\nstrides=(";

    for(dim = 0;dim < this->ndim - 1;dim++) specs << this->strides[dim] << ',' << ' ';
    specs << this->strides[dim] << ")\nndim=" << (uint16_t)this->ndim << "\nsize=" << this->size << '\n';

    return specs.str();
}

template <typename T> void NDArray<T>::print(bool specs, uint8_t dim, uint64_t data_index,
bool not_first, bool not_last) const
{
    uint32_t dim_idx;
    uint64_t stride;

    if(not_first) std::cout << std::string(dim, ' '); 
    std::cout << '[';

    if(dim == this->ndim - 1)
    {
        stride = this->strides[dim];

        for(dim_idx = 0;dim_idx < this->shape[dim] - 1;dim_idx++)
        {
            std::cout << this->data[data_index] << ',' << ' ';
            data_index += stride;
        }

        std::cout << this->data[data_index] << ']';
        if(not_last) std::cout << '\n';
        
        return;
    }

    this->print(specs, dim + 1, data_index, false, true);
    data_index += this->strides[dim];            

    for(dim_idx = 1;dim_idx < this->shape[dim] - 1;dim_idx++)
    {
        this->print(specs, dim + 1, data_index, true, true);
        data_index += this->strides[dim];
    }

    this->print(specs, dim + 1, data_index, true, false);

    std::cout << ']';
    
    if(!dim)
    {
        std::cout << '\n';
        if(specs) std::cout << '\n' << this->get_specs();
    }

    else if(not_last) std::cout << std::string(this->ndim - dim, '\n');
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
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T> const T& NDArray<T>::operator[](const NDIndex &ndindex) const
{
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T> NDArray<T> NDArray<T>::operator[](const SliceRanges &slice_ranges)
{
    NDArray<T> ndarray = this->shallow_copy();
    ndarray.slice_array(slice_ranges);
    return ndarray;
}

template <typename T> void NDArray<T>::operator+=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += value;
    }
}

template <typename T> void NDArray<T>::operator-=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= value;
    }
}

template <typename T> void NDArray<T>::operator*=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= value;
    }
}

template <typename T> void NDArray<T>::operator/=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= value;
    }
}

template <typename T> void NDArray<T>::shape_array(const Shape &shape)
{
    this->ndim = shape.size();
    uint64_t stride = 1;
    uint64_t size = shape[this->ndim - 1];

    this->strides = Strides(this->ndim);
    this->strides[this->ndim - 1] = stride;
    
    for(uint8_t dim = this->ndim - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        this->strides[dim] = stride;
        size *= shape[dim];
    }

    this->shape = Shape(shape);
    this->size = size;
}

template <typename T> void NDArray<T>::slice_array(const SliceRanges &slice_ranges)
{
    uint8_t nb_dims = slice_ranges.size() - 1;
    uint64_t stride = slice_ranges[nb_dims].step;
    uint64_t data_start = slice_ranges[nb_dims].start * this->strides[nb_dims];
    this->size = ceil_index((float64_t)(slice_ranges[nb_dims].end - slice_ranges[nb_dims].start) / slice_ranges[nb_dims].step);

    this->strides[nb_dims] = stride;
    this->shape[nb_dims] = size;

    for(uint8_t dim = nb_dims;dim-- >= 1;)
    {
        data_start += slice_ranges[dim].start * this->strides[dim];
        stride *= this->shape[dim + 1] * slice_ranges[dim].step;
        this->strides[dim] = stride;
        this->shape[dim] = ceil_index((float64_t)(slice_ranges[dim].end - slice_ranges[dim].start) / slice_ranges[dim].step);
        this->size *= this->shape[dim];
    }

    this->data += data_start;
}