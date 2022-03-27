
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

template <typename T, uint8_t N>
NDArray<T, N>& NDArray<T, N>::operator=(const NDArray<T, N> &ndarray)
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
    this->delete_data = true;

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }

    return *this;
}

template <typename T, uint8_t N>
NDArray<T, N>& NDArray<T, N>::operator=(NDArray<T, N> &&ndarray)
{
    if(this == &ndarray) { return *this; }

    if(this->delete_data) { delete[] this->data; }
    
    this->shape = std::move(ndarray.shape);
    this->strides = std::move(ndarray.strides);
    this->size = ndarray.size;
    this->delete_data = ndarray.delete_data;
    
    this->data = ndarray.data;
    ndarray.data = nullptr;

    return *this;
}

template <typename T, uint8_t N>
NDArray<T, N>::~NDArray()
{
    if(this->delete_data) { delete[] this->data; }
}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray() : data(nullptr), size(0), delete_data(true)
{}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(const Shape<N> &shape) : shape(shape), strides(Strides<N>()), delete_data(true)
{
    this->shape_array(shape);
    this->data = new T[size];
}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(const Shape<N> &shape, T fill_value) : NDArray<T, N>(shape)
{
    this->fill(fill_value);
}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(T *data, const Shape<N> &shape, const Strides<N> &strides,
uint64_t size, bool delete_data) : data(data), shape(shape), strides(strides),
size(size), delete_data(delete_data)
{}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(const NDArray<T, N> &ndarray) : NDArray<T, N>(new T[ndarray.size],
ndarray.get_shape(), ndarray.get_strides(), ndarray.get_size(), true)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = ndarray.data[idx];
    }
}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(T start, T end, T step) : NDArray<T, N>({ceil_index((end - start) / step)})
{
    uint64_t idx = 0;

    while(start < end)
    {
        this->data[idx] = start;
        start += step;
        idx++;
    }
}

template <typename T, uint8_t N>
NDArray<T, N>::NDArray(NDArray<T, N> &&ndarray) : data(ndarray.data), shape(std::move(ndarray.shape)),
strides(std::move(ndarray.strides)), size(ndarray.size), delete_data(ndarray.delete_data)
{
    ndarray.data = nullptr;
}

template <typename T, uint8_t N>
const T* NDArray<T, N>::get_data() const
{
    return this->data;
}

template <typename T, uint8_t N>
const Shape<N>& NDArray<T, N>::get_shape() const
{
    return this->shape;
}

template <typename T, uint8_t N>
const Strides<N>& NDArray<T, N>::get_strides() const
{
    return this->strides;
}

template <typename T, uint8_t N>
uint64_t NDArray<T, N>::get_size() const
{
    return this->size;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::does_delete_data()
{
    return this->delete_data;
}

template <typename T, uint8_t N>
void NDArray<T, N>::set_delete_data(bool delete_date)
{
    this->delete_data = delete_data;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::shallow_copy()
{
    return NDArray<T, N>(this->data, this->shape, this->strides, this->size, false);
}

template <typename T, uint8_t N>
void NDArray<T, N>::fill(T fill_value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] = fill_value;
    }
}

template <typename T, uint8_t N>
const NDArray<T, N> NDArray<T, N>::shallow_copy() const
{
    return NDArray<T, N>(this->data, this->shape, this->strides, this->size, false);
}

template <typename T, uint8_t N> template <uint8_t NN> NDArray<T, NN>
NDArray<T, N>::reshape(const Shape<NN> &shape) const
{
    NDArray<T, NN> ndarray(this->data, shape, Strides<NN>(), this->size, false);

    uint64_t stride = this->strides[N - 1];
    uint64_t size = shape[N - 1];

    ndarray.strides[N - 1] = stride;

    for(uint8_t dim = N - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        ndarray.strides[dim] = stride;
        size *= shape[dim];
    }

    assert(this->size == size);

    return ndarray;
}

template <typename T, uint8_t N>
uint64_t NDArray<T, N>::ravel_ndindex(const NDIndex<N> &ndindex) const
{
    uint64_t index = 0;
    uint8_t nb_dims = ndindex.size();

    for(uint8_t dim = 0;dim < nb_dims;dim++)
    {
        index += ndindex[dim] * this->strides[dim];
    }

    return index;
}

template <typename T, uint8_t N>
NDIndex<N> NDArray<T, N>::unravel_index(uint64_t index) const
{
    NDIndex<N> ndindex;

    for(uint8_t dim = 0;dim < N;dim++)
    {
        ndindex[dim] = index / this->strides[dim];
        index -= ndindex[dim] * this->strides[dim];
    }

    return ndindex;
}

template <typename T, uint8_t N> template <uint8_t NN>
NDArray<T, NN> NDArray<T, N>::shrink_dims() const
{
    NDArray<T, NN> ndarray(this->data, Shape<NN>(), Strides<NN>(), this->size, false);
    uint8_t new_dim = 0;

    for(uint8_t dim = 0;dim < N;dim++)
    {
        if(this->shape[dim] > 1)
        {
            ndarray.shape[new_dim] = this->shape[dim];
            ndarray.strides[new_dim] = this->strides[dim];
            new_dim++;
        }
    }

    return ndarray;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::dims_equal(const NDArray<T, N> &ndarray) const
{
    bool dims_equal = true;

    for(uint8_t dim = 0;dim < N && dims_equal;dim++)
    {
        dims_equal = (this->shape[dim] == ndarray.shape[dim]);
    }

    return dims_equal;
}

template <typename T, uint8_t N>
T NDArray<T, N>::max() const
{
    uint64_t max = *this->data;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        max = common::max<T, N>(max, this->data[idx]);
    }

    return max;
}

template <typename T, uint8_t N>
uint64_t NDArray<T, N>::index_max() const
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

template <typename T, uint8_t N>
NDIndex<N> NDArray<T, N>::ndindex_max() const
{
    return this->unravel_index(this->index_max());
}

template <typename T, uint8_t N>
T NDArray<T, N>::min() const
{
    uint64_t min = *this->data;

    for(uint64_t idx = 1;idx < this->size;idx++)
    {
        min = common::min<T, N>(min, this->data[idx]);
    }

    return min;
}

template <typename T, uint8_t N>
uint64_t NDArray<T, N>::index_min() const
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

template <typename T, uint8_t N>
NDIndex<N> NDArray<T, N>::ndindex_min() const
{
    return this->unravel_index(this->index_min());
}

template <typename T, uint8_t N>
std::string NDArray<T, N>::get_specs() const
{
    std::ostringstream specs;
    uint8_t dim;

    specs << "shape=(";
    for(dim = 0;dim < N - 1;dim++) specs << this->shape[dim] << ',' << ' ';
    specs << this->shape[dim] << ")\nstrides=(";

    for(dim = 0;dim < N - 1;dim++) specs << this->strides[dim] << ',' << ' ';
    specs << this->strides[dim] << ")\nndim=" << (uint16_t)N << "\nsize=" << this->size << '\n';

    return specs.str();
}

template <typename T, uint8_t N>
T& NDArray<T, N>::operator[](uint64_t index)
{
    return this->data[index];
}

template <typename T, uint8_t N>
const T& NDArray<T, N>::operator[](uint64_t index) const
{
    return this->data[index];
}

template <typename T, uint8_t N>
T& NDArray<T, N>::operator[](const NDIndex<N> &ndindex)
{
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T, uint8_t N>
const T& NDArray<T, N>::operator[](const NDIndex<N> &ndindex) const
{
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator[](const SliceRanges<N> &slice_ranges)
{
    NDArray<T, N> ndarray = this->shallow_copy();
    ndarray.slice_array(slice_ranges);
    return ndarray;
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator+=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += value;
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator-=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= value;
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator*=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= value;
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator/=(T value)
{
    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= value;
    }
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator+(T value) const
{
    NDArray<T, N> ndarray{new T[this->size], this->shape, this->strides, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] + value;
    }

    return ndarray;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator-(T value) const
{
    NDArray<T, N> ndarray{new T[this->size], this->shape, this->strides, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] - value;
    }

    return ndarray;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator*(T value) const
{
    NDArray<T, N> ndarray{new T[this->size], this->shape, this->strides, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] * value;
    }

    return ndarray;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator/(T value) const
{
    NDArray<T, N> ndarray{new T[this->size], this->shape, this->strides, this->size, true};

    for(uint64_t idx = 0;idx < ndarray.size;idx++)
    {
        ndarray.data[idx] = this->data[idx] / value;
    }

    return ndarray;
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator+=(const NDArray<T, N> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] += ndarray.data[idx];
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator-=(const NDArray<T, N> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] -= ndarray.data[idx];
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator*=(const NDArray<T, N> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] *= ndarray.data[idx];
    }
}

template <typename T, uint8_t N>
void NDArray<T, N>::operator/=(const NDArray<T, N> &ndarray)
{
    assert(this->dims_equal(ndarray));

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        this->data[idx] /= ndarray.data[idx];
    }
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator+(const NDArray<T, N> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T, N> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] + ndarray.data[idx];
    }

    return result_array;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator-(const NDArray<T, N> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T, N> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] - ndarray.data[idx];
    }

    return result_array;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator*(const NDArray<T, N> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T, N> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] * ndarray.data[idx];
    }

    return result_array;
}

template <typename T, uint8_t N>
NDArray<T, N> NDArray<T, N>::operator/(const NDArray<T, N> &ndarray) const
{
    assert(this->dims_equal(ndarray));
    NDArray<T, N> result_array(this->shape);

    for(uint64_t idx = 0;idx < this->size;idx++)
    {
        result_array.data[idx] = this->data[idx] / ndarray.data[idx];
    }

    return result_array;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator==(const NDArray<T, N> &ndarray) const
{
    bool eq = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && eq;idx++)
    {
        eq = (this->data[idx] == ndarray.data[idx]);
    }

    return eq;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator!=(const NDArray<T, N> &ndarray) const
{
    return !(*this == ndarray);
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator>=(const NDArray<T, N> &ndarray) const
{
    bool ge = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && ge;idx++)
    {
        ge = (this->data[idx] >= ndarray.data[idx]);
    }

    return ge;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator<=(const NDArray<T, N> &ndarray) const
{
    bool le = this->dims_equal(ndarray);

    for(uint64_t idx = 0;idx < this->size && le;idx++)
    {
        le = (this->data[idx] <= ndarray.data[idx]);
    }

    return le;
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator>(const NDArray<T, N> &ndarray) const
{
    return !(*this <= ndarray);
}

template <typename T, uint8_t N>
bool NDArray<T, N>::operator<(const NDArray<T, N> &ndarray) const
{
    return !(*this >= ndarray);
}

template <typename T, uint8_t N>
void NDArray<T, N>::print(bool print_specs, uint8_t dim, uint64_t data_index, bool not_first, bool not_last) const
{
    uint32_t dim_idx;
    uint64_t stride;

    if(not_first) std::cout << std::string(dim, ' '); 
    std::cout << '[';

    if(dim == N - 1)
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

    else if(not_last) std::cout << std::string(N - dim, '\n');
}

template <typename T, uint8_t N>
void NDArray<T, N>::shape_array(const Shape<N> &shape)
{
    uint64_t stride = 1;
    uint64_t size = (uint64_t)shape[N - 1];

    this->strides[N - 1] = stride;
    
    for(uint8_t dim = N - 1;dim-- > 0;)
    {
        stride *= shape[dim + 1];
        this->strides[dim] = stride;
        size *= shape[dim];
    }

    this->size = size;
}

template <typename T, uint8_t N>
void NDArray<T, N>::slice_array(const SliceRanges<N> &slice_ranges)
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