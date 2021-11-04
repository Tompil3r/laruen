
#include "laruen/ndarray/ndarray_core.h"
#include "laruen/ndarray/ndarray_typenames.h"
#include "laruen/ndarray/ndarray_utils.h"
#include "laruen/utils/range.h"
#include <cassert>
#include <ostream>
#include <cmath>
#include <cstdint>
#include <utility>

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


template <typename T> NDArray<T>& NDArray<T>::operator=(const NDArray<T> &ndarray)
{
    if(this == &ndarray)
    {
        return *this;
    }

    if(this->size != ndarray.size)
    {
        if(this->delete_data)
        {
            delete[] this->data;
        }
        this->data = new T[ndarray.size];
    }

    this->shape = ndarray.shape;
    this->strides = ndarray.strides;
    this->ndim = ndarray.ndim;
    this->size = ndarray.size;
    this->delete_data = true;

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }

    return *this;
}

template <typename T> NDArray<T>& NDArray<T>::operator=(NDArray<T> &&ndarray)
{
    if(this == &ndarray)
    {
        return *this;
    }

    if(this->delete_data)
    {
        delete[] this->data;
    }
    
    this->shape = std::move(ndarray.shape);
    this->strides = std::move(ndarray.strides);
    this->ndim = ndarray.ndim;
    this->size = ndarray.size;
    this->delete_data = ndarray.delete_data;
    
    this->data = ndarray.data;
    ndarray.data = nullptr;

    return *this;
}

template <typename T> NDArray<T>::~NDArray()
{
    if(this->delete_data)
    {
        delete[] this->data;
    }
}

template <typename T> NDArray<T>::NDArray() : data(nullptr), ndim(0), size(0), delete_data(true)
{
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
uint8_t ndim, uint64_t size, bool delete_data) : data(data), shape(shape), strides(strides),
ndim(ndim), size(size), delete_data(delete_data)
{
}

template <typename T> NDArray<T>::NDArray(const NDArray<T> &ndarray) : NDArray<T>(new T[ndarray.size],
ndarray.get_shape(), ndarray.get_strides(), ndarray.get_ndim(), ndarray.get_size(), true)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }
}

template <typename T> NDArray<T>::NDArray(T start, T stop, T step) : NDArray<T>({ceil_index((stop - start) / step)})
{
    uint64_t idx = 0;

    while(start < stop)
    {
        this->data[idx] = start;
        start += step;
        idx++;
    }
}

template <typename T> NDArray<T>::NDArray(NDArray &&ndarray) : data(ndarray.data), shape(std::move(ndarray.shape)),
strides(std::move(ndarray.strides)), ndim(ndarray.ndim), size(ndarray.size), delete_data(ndarray.delete_data)
{
    ndarray.data = nullptr;
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

template <typename T> bool NDArray<T>::dims_equal(const NDArray<T> &ndarray) const
{
    bool dims_equal = this->ndim == ndarray.ndim;

    for(uint8_t dim = 0;dim < this->ndim && dims_equal;dim++)
    {
        dims_equal = (this->shape[dim] == ndarray.shape[dim]);
    }

    return dims_equal;
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

template <typename T> NDArray<T> NDArray<T>::operator+(T value) const
{
    NDArray<T> ndarray{new T[this->size], this->shape, this->strides, this->ndim, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] + value;
    }

    return ndarray;
}

template <typename T> NDArray<T> NDArray<T>::operator-(T value) const
{
    NDArray<T> ndarray{new T[this->size], this->shape, this->strides, this->ndim, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] - value;
    }

    return ndarray;
}

template <typename T> NDArray<T> NDArray<T>::operator*(T value) const
{
    NDArray<T> ndarray{new T[this->size], this->shape, this->strides, this->ndim, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] * value;
    }

    return ndarray;
}

template <typename T> NDArray<T> NDArray<T>::operator/(T value) const
{
    NDArray<T> ndarray{new T[this->size], this->shape, this->strides, this->ndim, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] / value;
    }

    return ndarray;
}

template <typename T> void NDArray<T>::operator+=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += ndarray.data[idx];
    }
}

template <typename T> void NDArray<T>::operator-=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= ndarray.data[idx];
    }
}

template <typename T> void NDArray<T>::operator*=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= ndarray.data[idx];
    }
}

template <typename T> void NDArray<T>::operator/=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= ndarray.data[idx];
    }
}

template <typename T> NDArray<T> NDArray<T>::operator+(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] + ndarray.data[idx];
    }

    return result_array;
}

template <typename T> NDArray<T> NDArray<T>::operator-(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] - ndarray.data[idx];
    }

    return result_array;
}

template <typename T> NDArray<T> NDArray<T>::operator*(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] * ndarray.data[idx];
    }

    return result_array;
}

template <typename T> NDArray<T> NDArray<T>::operator/(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] / ndarray.data[idx];
    }

    return result_array;
}

template <typename T> void NDArray<T>::print(bool print_specs, uint8_t dim, uint64_t data_index,
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

    this->print(print_specs, dim + 1, data_index, false, true);
    data_index += this->strides[dim];            

    for(dim_idx = 1;dim_idx < this->shape[dim] - 1;dim_idx++)
    {
        this->print(print_specs, dim + 1, data_index, true, true);
        data_index += this->strides[dim];
    }

    this->print(print_specs, dim + 1, data_index, true, false);

    std::cout << ']';
    
    if(!dim)
    {
        std::cout << '\n';
        if(print_specs) std::cout << '\n' << this->get_specs();
    }

    else if(not_last) std::cout << std::string(this->ndim - dim, '\n');
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

template <typename T> Shape NDArray<T>::broadcast_shapes(const NDArray<T> &ndarray) const
{
    Shape shape;
    uint8_t min_dims;
    uint8_t shape_ndim;
    bool broadcastable = true;
    uint32_t tdim;
    uint32_t odim;

    if(this->ndim > ndarray.ndim)
    {
        shape = this->shape;
        shape_ndim = this->ndim;
        min_dims = ndarray.ndim;
    }
    else
    {
        shape = ndarray.shape;
        shape_ndim = ndarray.ndim;
        min_dims = this->ndim;
    }

    for(uint8_t dim = 1;dim <= min_dims && (tdim = this->shape[this->ndim - dim], odim = ndarray.shape[ndarray.ndim - dim],
    broadcastable = (tdim == odim || tdim == 1 || odim == 1));dim++)
    {
        shape[shape_ndim - dim] = (tdim > odim ? tdim : odim);
    }

    if(!broadcastable) shape.clear();
    return shape;
}

template <typename T> bool NDArray<T>::output_broadcastable(const NDArray<T> &ndarray) const
{
    bool broadcastable = this->ndim <= ndarray.ndim;
    uint32_t odim;

    for(uint8_t dim = 1;dim <= ndarray.ndim && (odim = ndarray.shape[ndarray.ndim - dim],
    broadcastable = (this->shape[this->ndim - dim] == odim || odim == 1));dim++);

    return broadcastable;
}