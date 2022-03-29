
#include "laruen/ndarray/ndarray.h"
#include "laruen/ndarray/ndarray_types.h"
#include "laruen/ndarray/ndarray_utils.h"
#include "laruen/ndarray/ndarray_lib.h"
#include "laruen/utils/range.h"
#include "laruen/math/common.h"
#include <cassert>
#include <ostream>
#include <cstdint>
#include <utility>
#include <stdexcept>

using namespace laruen::ndarray;
using namespace laruen::ndarray::utils;
using namespace laruen::math;

template <typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T> &ndarray) {
    if(this == &ndarray) {
        return *this;
    }

    if(this->size != ndarray.size) {
        if(this->free_mem) {
            delete[] this->data;
        }
        this->data = new T[ndarray.size];
    }

    this->shape = ndarray.shape;
    this->strides = ndarray.strides;
    this->size = ndarray.size;
    this->ndim = ndarray.ndim;
    this->free_mem = true;

    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] = ndarray.data[i];
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T> &&ndarray) {
    if(this == &ndarray) {
        return *this;
    }

    if(this->free_mem) {
        delete[] this->data;
    }
    
    this->shape = std::move(ndarray.shape);
    this->strides = std::move(ndarray.strides);
    this->size = ndarray.size;
    this->ndim = ndarray.ndim;
    this->free_mem = ndarray.free_mem;
    
    this->data = ndarray.data;
    ndarray.data = nullptr;

    return *this;
}

template <typename T>
NDArray<T>::~NDArray() {
    if(this->free_mem) {
        delete[] this->data;
    }
}

template <typename T>
NDArray<T>::NDArray() : data(nullptr), size(0), ndim(0), free_mem(true) {}

template <typename T>
NDArray<T>::NDArray(const Shape &shape) : shape(shape), strides(Strides()), ndim(shape.size()), free_mem(true) {
    this->shape_array(shape);
    this->data = new T[size];
}

template <typename T>
NDArray<T>::NDArray(const Shape &shape, T fill) : NDArray<T>(shape) {
    this->fill(fill);
}

template <typename T>
NDArray<T>::NDArray(T *data, const Shape &shape, const Strides &strides,
uint64_t size, uint8_t ndim, bool free_mem) : data(data), shape(shape), strides(strides),
size(size), ndim(ndim), free_mem(free_mem) {}

template <typename T>
NDArray<T>::NDArray(const NDArray<T> &ndarray) : NDArray<T>(new T[ndarray.size],
ndarray.get_shape(), ndarray.get_strides(), ndarray.get_size(), ndim, true)
{
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] = ndarray.data[i];
    }
}

template <typename T>
NDArray<T>::NDArray(T start, T end, T step) : NDArray<T>({ceil_index((end - start) / step)}) {
    uint64_t i = 0;

    while(start < end) {
        this->data[i] = start;
        start += step;
        i++;
    }
}

template <typename T>
NDArray<T>::NDArray(NDArray<T> &&ndarray) : data(ndarray.data), shape(std::move(ndarray.shape)),
strides(std::move(ndarray.strides)), size(ndarray.size), ndim(ndarray.ndim), free_mem(ndarray.free_mem)
{
    ndarray.data = nullptr;
}

template <typename T>
NDArray<T> NDArray<T>::shallow_copy() {
    return NDArray<T>(this->data, this->shape, this->strides, this->size, this->ndim, false);
}

template <typename T>
void NDArray<T>::fill(T fill) {
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] = fill;
    }
}

template <typename T>
const NDArray<T> NDArray<T>::shallow_copy() const {
    return NDArray<T>(this->data, this->shape, this->strides, this->size, this->ndim, false);
}

template <typename T>
NDArray<T> NDArray<T>::reshape(const Shape &shape) {
    uint8_t ndim = shape.size();
    NDArray<T> ndarray(this->data, shape, Strides(), this->size, ndim, false);

    uint64_t stride = this->strides[ndim - 1];
    uint64_t size = shape[ndim - 1];

    ndarray.strides[ndim - 1] = stride;

    for(uint8_t dim = ndim - 1;dim-- > 0;) {
        stride *= shape[dim + 1];
        ndarray.strides[dim] = stride;
        size *= shape[dim];
    }

    assert(this->size == size);

    return ndarray;
}

template <typename T>
uint64_t NDArray<T>::ravel_ndindex(const NDIndex &ndindex) const {
    uint64_t index = 0;
    uint8_t ndim = ndindex.size();

    for(uint8_t dim = 0;dim < ndim;dim++) {
        index += ndindex[dim] * this->strides[dim];
    }

    return index;
}

template <typename T>
NDIndex NDArray<T>::unravel_index(uint64_t index) const {
    NDIndex ndindex;

    for(uint8_t dim = 0;dim < this->ndim;dim++) {
        ndindex[dim] = index / this->strides[dim];
        index -= ndindex[dim] * this->strides[dim];
    }

    return ndindex;
}

template <typename T>
NDArray<T> NDArray<T>::shrink_dims() const {
    NDArray<T> ndarray(this->data, Shape(), Strides(), this->size, this->ndim, false);
    uint8_t new_ndim = 0;

    for(uint8_t dim = 0;dim < this->ndim;dim++) {
        if(this->shape[dim] > 1) {
            ndarray.shape.push_back(this->shape[dim]);
            ndarray.strides.push_back(this->strides[dim]);
            new_ndim++;
        }
    }

    ndarray.ndim = new_ndim;

    return ndarray;
}

template <typename T>
bool NDArray<T>::eq_dims(const NDArray<T> &ndarray) const {
    bool eq_dims = this->ndim == ndarray.ndim;

    for(uint8_t dim = 0;dim < this->ndim && eq_dims;dim++) {
        eq_dims = (this->shape[dim] == ndarray.shape[dim]);
    }

    return eq_dims;
}

template <typename T>
T NDArray<T>::max() const {
    uint64_t max = *this->data;

    for(uint64_t i = 1;i < this->size;i++) {
        max = common::max(max, this->data[i]);
    }

    return max;
}

template <typename T>
uint64_t NDArray<T>::index_max() const {
    uint64_t max = *this->data;
    uint64_t index_max = 0;

    for(uint64_t i = 1;i < this->size;i++) {
        if(this->data[i] > max) {
            max = this->data[i];
            index_max = i;
        }
    }

    return index_max;
}

template <typename T>
NDIndex NDArray<T>::ndindex_max() const {
    return this->unravel_index(this->index_max());
}

template <typename T>
T NDArray<T>::min() const {
    uint64_t min = *this->data;

    for(uint64_t i = 1;i < this->size;i++) {
        min = common::min(min, this->data[i]);
    }

    return min;
}

template <typename T>
uint64_t NDArray<T>::index_min() const {
    uint64_t min = *this->data;
    uint64_t index_min = 0;

    for(uint64_t i = 1;i < this->size;i++) {
        if(this->data[i] < min) {
            min = this->data[i];
            index_min = i;
        }
    }

    return index_min;
}

template <typename T>
NDIndex NDArray<T>::ndindex_min() const {
    return this->unravel_index(this->index_min());
}

template <typename T>
std::string NDArray<T>::info() const {
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
T& NDArray<T>::operator[](const NDIndex &ndindex) {
    return this->data[this->ravel_ndindex(ndindex)];
}

template <typename T>
const T& NDArray<T>::operator[](const NDIndex &ndindex) const {
    return this->data[this->ravel_ndindex(ndindex)];
}

/*
template <typename T>
const NDArray<T> NDArray<T>::operator[](const SliceRanges &slice_ranges) const {
    NDArray<T> ndarray = this->shallow_copy();
    ndarray.slice_array(slice_ranges);
    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator[](const SliceRanges &slice_ranges) {
    NDArray<T> ndarray = this->shallow_copy();
    ndarray.slice_array(slice_ranges);
    return ndarray;
}
*/

template <typename T>
NDArray<T>& NDArray<T>::operator+=(T value) {
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] += value;
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator-=(T value) {
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] -= value;
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator*=(T value) {
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] *= value;
    }
    
    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator/=(T value) {
    for(uint64_t i = 0;i < this->size;i++) {
        this->data[i] /= value;
    }

    return *this;
}

template <typename T>
NDArray<T> NDArray<T>::operator+(T value) const {
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, this->ndim, true);

    for(uint64_t i = 0;i < ndarray.size;i++) {
        ndarray.data[i] = this->data[i] + value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator-(T value) const {
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t i = 0;i < ndarray.size;i++) {
        ndarray.data[i] = this->data[i] - value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator*(T value) const {
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t i = 0;i < ndarray.size;i++) {
        ndarray.data[i] = this->data[i] * value;
    }

    return ndarray;
}

template <typename T>
NDArray<T> NDArray<T>::operator/(T value) const {
    NDArray<T> ndarray(new T[this->size], this->shape, this->strides, this->size, true);

    for(uint64_t i = 0;i < ndarray.size;i++) {
        ndarray.data[i] = this->data[i] / value;
    }

    return ndarray;
}

template <typename T>
NDArray<T>& NDArray<T>::operator+=(const NDArray<T> &ndarray) {
    if(!ndarray::eq_dims(this->shape, ndarray::d_broadcast(this->shape, ndarray.shape))) {
        throw std::invalid_argument("shapes cannot be broadcasted");
    }

    uint64_t size_ratio = this->size / ndarray.size;
    uint64_t d_idx = 0; // destination index

    for(uint64_t br_idx = 0;br_idx < size_ratio;br_idx++) { // br_idx - broadcasting index
        for(uint64_t s_idx = 0;s_idx < ndarray.size;s_idx++) { // s_idx - source index
            this->data[d_idx] += ndarray[s_idx];
            d_idx++;
        }
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator-=(const NDArray<T> &ndarray) {
    if(!ndarray::eq_dims(this->shape, ndarray::d_broadcast(this->shape, ndarray.shape))) {
        throw std::invalid_argument("shapes cannot be broadcasted");
    }

    uint64_t size_ratio = this->size / ndarray.size;
    uint64_t d_idx = 0; // destination index

    for(uint64_t br_idx = 0;br_idx < size_ratio;br_idx++) { // br_idx - broadcasting index
        for(uint64_t s_idx = 0;s_idx < ndarray.size;s_idx++) { // s_idx - source index
            this->data[d_idx] -= ndarray[s_idx];
            d_idx++;
        }
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator*=(const NDArray<T> &ndarray) {
    if(!ndarray::eq_dims(this->shape, ndarray::d_broadcast(this->shape, ndarray.shape))) {
        throw std::invalid_argument("shapes cannot be broadcasted");
    }

    uint64_t size_ratio = this->size / ndarray.size;
    uint64_t d_idx = 0; // destination index

    for(uint64_t br_idx = 0;br_idx < size_ratio;br_idx++) { // br_idx - broadcasting index
        for(uint64_t s_idx = 0;s_idx < ndarray.size;s_idx++) { // s_idx - source index
            this->data[d_idx] *= ndarray[s_idx];
            d_idx++;
        }
    }

    return *this;
}

template <typename T>
NDArray<T>& NDArray<T>::operator/=(const NDArray<T> &ndarray) {
    if(!ndarray::eq_dims(this->shape, ndarray::d_broadcast(this->shape, ndarray.shape))) {
        throw std::invalid_argument("shapes cannot be broadcasted");
    }

    uint64_t size_ratio = this->size / ndarray.size;
    uint64_t d_idx = 0; // destination index

    for(uint64_t br_idx = 0;br_idx < size_ratio;br_idx++) { // br_idx - broadcasting index
        for(uint64_t s_idx = 0;s_idx < ndarray.size;s_idx++) { // s_idx - source index
            this->data[d_idx] /= ndarray[s_idx];
            d_idx++;
        }
    }

    return *this;
}

template <typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T> &ndarray) const {
    assert(this->eq_dims(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t i = 0;i < this->size;i++) {
        result_array.data[i] = this->data[i] + ndarray.data[i];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T> &ndarray) const {
    assert(this->eq_dims(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t i = 0;i < this->size;i++) {
        result_array.data[i] = this->data[i] - ndarray.data[i];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T> &ndarray) const {
    assert(this->eq_dims(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t i = 0;i < this->size;i++) {
        result_array.data[i] = this->data[i] * ndarray.data[i];
    }

    return result_array;
}

template <typename T>
NDArray<T> NDArray<T>::operator/(const NDArray<T> &ndarray) const {
    assert(this->eq_dims(ndarray)); // needs broadcasting
    NDArray<T> result_array(this->shape);

    for(uint64_t i = 0;i < this->size;i++) {
        result_array.data[i] = this->data[i] / ndarray.data[i];
    }

    return result_array;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator==(const NDArray<T> &ndarray) const {
    bool eq = this->eq_dims(ndarray);

    for(uint64_t i = 0;i < this->size && eq;i++) {
        eq = (this->data[i] == ndarray.data[i]);
    }

    return eq;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator!=(const NDArray<T> &ndarray) const {
    return !(*this == ndarray);
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator>=(const NDArray<T> &ndarray) const {
    bool ge = this->eq_dims(ndarray);

    for(uint64_t i = 0;i < this->size && ge;i++) {
        ge = (this->data[i] >= ndarray.data[i]);
    }

    return ge;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator<=(const NDArray<T> &ndarray) const {
    bool le = this->eq_dims(ndarray);

    for(uint64_t i = 0;i < this->size && le;i++) {
        le = (this->data[i] <= ndarray.data[i]);
    }

    return le;
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator>(const NDArray<T> &ndarray) const {
    return !(*this <= ndarray);
}

template <typename T> // allow comparison of different types
bool NDArray<T>::operator<(const NDArray<T> &ndarray) const {
    return !(*this >= ndarray);
}

template <typename T>
void NDArray<T>::shape_array(const Shape &shape) {
    uint64_t stride = 1;
    uint64_t size = shape[this->ndim - 1];
    this->strides.resize(this->ndim);

    this->strides[this->ndim - 1] = stride;
    
    for(uint8_t dim = this->ndim - 1;dim-- > 0;) {
        stride *= shape[dim + 1];
        this->strides[dim] = stride;
        size *= shape[dim];
    }

    this->size = size;
}

template <typename T>
void NDArray<T>::slice_array(const SliceRanges &slice_ranges) {

    uint8_t ndim = slice_ranges.size() - 1;
    uint64_t stride = slice_ranges[ndim].step;
    uint64_t data_start = slice_ranges[ndim].start * this->strides[ndim];
    this->size = ceil_index((float64_t)(slice_ranges[ndim].end - slice_ranges[ndim].start) / slice_ranges[ndim].step);

    this->strides[ndim] = stride;
    this->shape[ndim] = size;

    for(uint8_t dim = ndim;dim-- >= 1;) {
        data_start += slice_ranges[dim].start * this->strides[dim];
        stride *= this->shape[dim + 1] * slice_ranges[dim].step;
        this->strides[dim] = stride;
        this->shape[dim] = ceil_index((float64_t)(slice_ranges[dim].end - slice_ranges[dim].start) / slice_ranges[dim].step);
        this->size *= this->shape[dim];
    }

    this->data += data_start;
}

template <typename T>
void NDArray<T>::str_(std::string &str, uint8_t dim, uint64_t data_index, bool not_first, bool not_last) const {
    uint64_t dim_idx;
    uint64_t stride;

    if(not_first) {
        str += std::string(dim, ' ');
    }

    str.push_back('[');

    if(dim == this->ndim - 1) {
        stride = this->strides[dim];

        if(this->shape[dim]) {
            for(dim_idx = 0;dim_idx < this->shape[dim] - 1;dim_idx++) {
                str += std::to_string(this->data[data_index]);
                str.push_back(',');
                str.push_back(' ');
                data_index += stride;
            }

            str += std::to_string(this->data[data_index]);
        }

        str.push_back(']');
        if(not_last) {
            str.push_back('\n');
        }
        
        return;
    }

    if(this->shape[dim]) {
        this->str_(str, dim + 1, data_index, false, this->shape[dim] > 1);
        data_index += this->strides[dim];

        for(dim_idx = 1;dim_idx < this->shape[dim] - 1;dim_idx++) {
            this->str_(str, dim + 1, data_index, true, true);
            data_index += this->strides[dim];
        }
    }

    if(this->shape[dim] > 1) {
        this->str_(str, dim + 1, data_index, true, false);
    }
    
    str.push_back(']');
    
    if(!dim) {
        str.push_back('\n');
    }

    else if(not_last) {
        str += std::string(this->ndim - dim, '\n');
    }
}