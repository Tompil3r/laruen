
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/ndarray_types.h"
#include "laruen/ndarray/ndarray_utils.h"
#include "laruen/utils/range.h"
#include "laruen/math/common.h"
#include <cassert>
#include <ostream>
#include <cstdint>
#include <utility>

using namespace laruen::ndarray;
using namespace laruen::ndarray::utils;
using namespace laruen::math;

template <typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T> &ndarray)
{
    if(this == &ndarray) { return *this; }

    if(this->size != ndarray.size)
    {
        if(this->delete_data) { delete[] this->data; }
        this->data = new T[ndarray.size];
    }

    this->shape = ndarray.shape;
    this->strides = ndarray.strides;
    this->size = ndarray.size;
    this->ndim = ndarray.ndim;
    this->delete_data = true;

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T> &&ndarray)
{
    if(this == &ndarray) { return *this; }

    if(this->delete_data) { delete[] this->data; }
    
    this->shape = std::move(ndarray.shape);
    this->strides = std::move(ndarray.strides);
    this->size = ndarray.size;
    this->ndim = ndarray.ndim;
    this->delete_data = ndarray.delete_data;
    
    this->data = ndarray.data;
    ndarray.data = nullptr;

    return *this;
}

template <typename T>
NDArray<T>::~NDArray()
{
    if(this->delete_data) { delete[] this->data; }
}

template <typename T>
NDArray<T>::NDArray() : data(nullptr), size(0), ndim(0), delete_data(true)
{}

template <typename T>
NDArray<T>::NDArray(const Shape &shape) : shape(shape), strides(Strides()), ndim(shape.size()), delete_data(true)
{
    this->shape_array(shape);
    this->data = new T[size];
}

template <typename T>
NDArray<T>::NDArray(const Shape &shape, T fill_value) : NDArray<T>(shape)
{
    this->fill(fill_value);
}

template <typename T>
NDArray<T>::NDArray(T *data, const Shape &shape, const Strides &strides,
uint64_t size, uint8_t ndim, bool delete_data) : data(data), shape(shape), strides(strides),
size(size), ndim(ndim), delete_data(delete_data)
{}

template <typename T>
NDArray<T>::NDArray(const NDArray<T> &ndarray) : NDArray<T>(new T[ndarray.size],
ndarray.get_shape(), ndarray.get_strides(), ndarray.get_size(), ndim, true)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }
}

template <typename T>
NDArray<T>::NDArray(T start, T end, T step) : NDArray<T>({ceil_index((end - start) / step)})
{
    uint64_t idx = 0;

    while(start < end)
    {
        this->data[idx] = start;
        start += step;
        idx++;
    }
}

template <typename T>
NDArray<T>::NDArray(NDArray<T> &&ndarray) : data(ndarray.data), shape(std::move(ndarray.shape)),
strides(std::move(ndarray.strides)), size(ndarray.size), ndim(ndarray.ndim), delete_data(ndarray.delete_data)
{
    ndarray.data = nullptr;
}

template <typename T>
const T* NDArray<T>::get_data() const
{
    return this->data;
}

template <typename T>
const Shape& NDArray<T>::get_shape() const
{
    return this->shape;
}

template <typename T>
const Strides& NDArray<T>::get_strides() const
{
    return this->strides;
}

template <typename T>
uint64_t NDArray<T>::get_size() const
{
    return this->size;
}

template <typename T>
bool NDArray<T>::does_delete_data()
{
    return this->delete_data;
}

template <typename T>
void NDArray<T>::set_delete_data(bool delete_date)
{
    this->delete_data = delete_data;
}

template <typename T>
NDArray<T> NDArray<T>::shallow_copy()
{
    return NDArray<T>(this->data, this->shape, this->strides, this->size, this->ndim, false);
}

template <typename T>
void NDArray<T>::fill(T fill_value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = fill_value;
    }
}

template <typename T>
const NDArray<T> NDArray<T>::shallow_copy() const
{
    return NDArray<T>(this->data, this->shape, this->strides, this->size, this->ndim, false);
}

template <typename T>
NDArray<T> NDArray<T>::reshape(const Shape &shape) const
{
    uint8_t ndim = shape.size();
    NDArray<T> ndarray(this->data, shape, Strides(), this->size, ndim, false);

    uint64_t stride = this->strides[ndim - 1];
    uint64_t size = shape[ndim - 1];

    ndarray.strides[ndim - 1] = stride;

    for(uint8_t dim = ndim - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        ndarray.strides[dim] = stride;
        size *= shape[dim];
    }

    assert(this->size == size);

    return ndarray;
}

template <typename T>
uint64_t NDArray<T>::ravel_ndindex(const NDIndex &ndindex) const
{
    uint64_t index = 0;
    uint8_t ndim = ndindex.size();

    for(uint8_t dim = 0;dim < ndim;dim++)
    {
        index += ndindex[dim] * this->strides[dim];
    }

    return index;
}

template <typename T>
NDIndex NDArray<T>::unravel_index(uint64_t index) const
{
    NDIndex ndindex;

    for(uint8_t dim = 0;dim < this->ndim;dim++)
    {
        ndindex[dim] = index / this->strides[dim];
        index -= ndindex[dim] * this->strides[dim];
    }

    return ndindex;
}

template <typename T>
NDArray<T> NDArray<T>::shrink_dims() const
{
    NDArray<T> ndarray(this->data, Shape(), Strides(), this->size, this->ndim, false);
    uint8_t new_ndim = 0;

    for(uint8_t dim = 0;dim < this->ndim;dim++)
    {
        if(this->shape[dim] > 1)
        {
            ndarray.shape.push_back(this->shape[dim]);
            ndarray.strides.push_back(this->strides[dim]);
            new_ndim++;
        }
    }

    ndarray.ndim = new_ndim;

    return ndarray;
}

template <typename T>
bool NDArray<T>::dims_equal(const NDArray<T> &ndarray) const
{
    bool dims_equal = this->ndim == ndarray.ndim;

    for(uint8_t dim = 0;dim < this->ndim && dims_equal;dim++)
    {
        dims_equal = (this->shape[dim] == ndarray.shape[dim]);
    }

    return dims_equal;
}

template <typename T>
T NDArray<T>::max() const
{
    uint64_t max = *this->data;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        max = common::max(max, this->data[idx]);
    }

    return max;
}

template <typename T>
uint64_t NDArray<T>::index_max() const
{
    uint64_t max = *this->data;
    uint64_t index_max = 0;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        if(this->data[idx] > max)
        {
            max = this->data[idx];
            index_max = idx;
        }
    }

    return index_max;
}

template <typename T>
NDIndex NDArray<T>::ndindex_max() const
{
    return this->unravel_index(this->index_max());
}

template <typename T>
T NDArray<T>::min() const
{
    uint64_t min = *this->data;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        min = common::min(min, this->data[idx]);
    }

    return min;
}

template <typename T>
uint64_t NDArray<T>::index_min() const
{
    uint64_t min = *this->data;
    uint64_t index_min = 0;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        if(this->data[idx] < min)
        {
            min = this->data[idx];
            index_min = idx;
        }
    }

    return index_min;
}

template <typename T>
NDIndex NDArray<T>::ndindex_min() const
{
    return this->unravel_index(this->index_min());
}

template <typename T>
std::string NDArray<T>::get_specs() const
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

template <typename T>
T& NDArray<T>::operator[](uint64_t index)
{
    return this->data[index];
}

template <typename T>
const T& NDArray<T>::operator[](uint64_t index) const
{
    return this->data[index];
}

template <typename T>
T& NDArray<T>::operator[](const NDIndex &ndindex)
{
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T>
const T& NDArray<T>::operator[](const NDIndex &ndindex) const
{
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T>
NDArray<T> NDArray<T>::operator[](const SliceRanges &slice_ranges)
{
    NDArray<T> ndarray = this->shallow_copy();
    ndarray.slice_array(slice_ranges);
    return ndarray;
}

template <typename T>
void NDArray<T>::operator+=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += value;
    }
}

template <typename T>
void NDArray<T>::operator-=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= value;
    }
}

template <typename T>
void NDArray<T>::operator*=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= value;
    }
}

template <typename T>
void NDArray<T>::operator/=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= value;
    }
}

template <typename T>
NDArray<T> NDArray<T>::operator+(T value) const
{
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, this->ndim, true);

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] + value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator-(T value) const
{
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] - value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator*(T value) const
{
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] * value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator/(T value) const
{
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] / value;
    }

    return ndarray;
}

template <typename T>
void NDArray<T>::operator+=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray)); // needs broadcasting

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += ndarray.data[idx];
    }
}

template <typename T>
void NDArray<T>::operator-=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray)); // needs broadcasting

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= ndarray.data[idx];
    }
}

template <typename T>
void NDArray<T>::operator*=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray)); // needs broadcasting

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= ndarray.data[idx];
    }
}

template <typename T>
void NDArray<T>::operator/=(const NDArray<T> &ndarray)
{
    assert(this->dims_equal(ndarray)); // needs broadcasting

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= ndarray.data[idx];
    }
}

template <typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray));  // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] + ndarray.data[idx];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] - ndarray.data[idx];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] * ndarray.data[idx];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator/(const NDArray<T> &ndarray) const
{
    assert(this->dims_equal(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] / ndarray.data[idx];
    }

    return result_array;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator==(const NDArray<T> &ndarray) const
{
    bool eq = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && eq;idx++)
    {
        eq = (this->data[idx] == ndarray.data[idx]);
    }

    return eq;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator!=(const NDArray<T> &ndarray) const
{
    return !(*this == ndarray);
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator>=(const NDArray<T> &ndarray) const
{
    bool ge = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && ge;idx++)
    {
        ge = (this->data[idx] >= ndarray.data[idx]);
    }

    return ge;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator<=(const NDArray<T> &ndarray) const
{
    bool le = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && le;idx++)
    {
        le = (this->data[idx] <= ndarray.data[idx]);
    }

    return le;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator>(const NDArray<T> &ndarray) const
{
    return !(*this <= ndarray);
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator<(const NDArray<T> &ndarray) const
{
    return !(*this >= ndarray);
}

template <typename T>
void NDArray<T>::print(bool print_specs, uint8_t dim, uint64_t data_index, bool not_first, bool not_last) const
{
    uint64_t dim_idx;
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
        if(not_last) { std::cout << '\n'; }
        
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

template <typename T>
void NDArray<T>::shape_array(const Shape &shape)
{
    uint64_t stride = 1;
    uint64_t size = (uint64_t)shape[this->ndim - 1];
    this->strides.resize(this->ndim);

    this->strides[this->ndim - 1] = stride;
    
    for(uint8_t dim = this->ndim - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        this->strides[dim] = stride;
        size *= shape[dim];
    }

    this->size = size;
}

template <typename T>
void NDArray<T>::slice_array(const SliceRanges &slice_ranges)
{
    uint8_t ndim = slice_ranges.size() - 1;
    uint64_t stride = slice_ranges[ndim].step;
    uint64_t data_start = slice_ranges[ndim].start * this->strides[ndim];
    this->size = ceil_index((float64_t)(slice_ranges[ndim].end - slice_ranges[ndim].start) / slice_ranges[ndim].step);

    this->strides[ndim] = stride;
    this->shape[ndim] = size;

    for(uint8_t dim = ndim;dim-- >= 1;)
    {
        data_start += slice_ranges[dim].start * this->strides[dim];
        stride *= this->shape[dim + 1] * slice_ranges[dim].step;
        this->strides[dim] = stride;
        this->shape[dim] = ceil_index((float64_t)(slice_ranges[dim].end - slice_ranges[dim].start) / slice_ranges[dim].step);
        this->size *= this->shape[dim];
    }

    this->data += data_start;
}

/*
template <typename T, uint8_t N>
Shape NDArray<T, N>::broadcast_shapes(const NDArray<T, N> &ndarray) const
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
*/

/*
template <typename T, uint8_t N>
bool NDArray<T, N>::output_broadcastable(const NDArray<T, N> &ndarray) const
{
    bool broadcastable = this->ndim <= ndarray.ndim;
    uint32_t odim;

    for(uint8_t dim = 1;dim <= ndarray.ndim && (odim = ndarray.shape[ndarray.ndim - dim],
    broadcastable = (this->shape[this->ndim - dim] == odim || odim == 1));dim++);

    return broadcastable;
}
*/