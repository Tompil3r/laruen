
#include "src/ndlib/ndarray.h"
#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/ndarray_utils.h"
#include "src/ndlib/ndlib.h"
#include "src/ndlib/nditerator.h"
#include "src/utils/range.h"
#include "src/math/common.h"
#include <cassert>
#include <ostream>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <cmath>

using namespace laruen;
using namespace laruen::ndlib::utils;
using namespace laruen::math;

namespace laruen::ndlib {

    template <typename T, bool C>
    NDArray<T, C>::~NDArray() {
        if(this->m_free_mem) {
            delete[] this->m_data;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray() : ArrayBase(), m_data(nullptr) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(uint8_t ndim, T *data, bool free_mem, uint64_t size)
    : ArrayBase(ndim, free_mem, size), m_data(data) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape)
    : ArrayBase(shape), m_data(new T[this->m_size]) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape, T value) : NDArray<T, C>(shape) {
        this->fill(value);
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const ArrayBase &base)
    : ArrayBase(base), m_data(data) {}
    
    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const ArrayBase &base, bool free_mem)
    : ArrayBase(base, free_mem), m_data(data) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const NDArray<T, C> &ndarray)
    : NDArray<T, C>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(NDArray<T, C> &&ndarray)
    : ArrayBase(std::move(ndarray)), m_data(ndarray.m_data)
    {
        ndarray.m_data = nullptr;
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Range<T> &range)
    : NDArray<T, C>(Shape{ceil_index((range.end - range.start) / range.step)}) {
        T value = range.start;

        for(uint64_t i = 0;i < this->m_size;i++) {
            this->m_data[i] = value;
            value += range.step;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(NDArray<T, C> &ndarray, const Axes &axes)
    : ArrayBase(axes.size(), false), m_data(ndarray.m_data)
    {
        this->m_size = this->m_ndim > 0;
        uint8_t axis;

        for(uint8_t i = 0;i < this->m_ndim;i++) {
            axis = axes[i];
            this->m_shape[i] = ndarray.m_shape[axis];
            this->m_strides[i] = ndarray.m_strides[axis];
            this->m_size *= this->m_shape[i];
        }
    }

    template <typename T, bool C> template <bool C2>
    NDArray<T, C>::NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges)
    : NDArray<T, C>(ndarray.m_data, ndarray, false)
    {
        static_assert(!C, "cannot create sliced array with non-sliced type");

        uint8_t ndim = ranges.size();
        float64_t size_ratio = 1;

        for(uint8_t dim = 0;dim < ndim;dim++) {
            size_ratio *= this->m_shape[dim];
            this->m_data += ranges[dim].start * this->m_strides[dim];
            this->m_strides[dim] = this->m_strides[dim] * ranges[dim].step;
            this->m_shape[dim] = ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
            size_ratio /= this->m_shape[dim];
        }

        this->m_size /= size_ratio;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>::NDArray(const NDArray<T2, C2> &ndarray, bool free_mem)
    : NDArray<T, C>(ndarray.m_data, ndarray, free_mem) {}

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>::NDArray(const NDArray<T2, C2> &ndarray)
    : NDArray<T, C>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>::NDArray(NDArray<T2, C2> &&ndarray)
    : ArrayBase(std::move(ndarray)), m_data(new T[ndarray.m_size])
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::operator=(const NDArray<T, C> &ndarray) {
        if(this == &ndarray) {
            return *this;
        }

        if(this->m_size != ndarray.m_size) {
            if(this->m_free_mem) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T, C> &&ndarray) {
        if(this == &ndarray) {
            return *this;
        }

        if(this->m_free_mem) {
            delete[] this->m_data;
        }
        
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = ndarray.m_free_mem;
        
        this->m_data = ndarray.m_data;
        ndarray.m_data = nullptr;

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator=(const NDArray<T2, C2> &ndarray) {
        if(this->m_size != ndarray.m_size) {
            if(this->m_free_mem) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T2, C2> &&ndarray) {
        this->m_data = new T[ndarray.m_size];
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    void NDArray<T, C>::copy_data_from(const NDArray<T2, C2> &ndarray) {
        NDIterator to(*this);
        ConstNDIterator from(ndarray);

        for(uint64_t i = 0;i < this->m_size;i++) {
            to.next() = from.next();
        }
    }

    template <typename T, bool C>
    void NDArray<T, C>::fill(T value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() = value;
        }
    }

    template <typename T, bool C>
    T NDArray<T, C>::max() const {
        ConstNDIterator iter(*this);
        T max = iter.next();

        for(uint64_t i = 1;i < this->m_size;i++) {
            max = common::max(max, iter.next());
        }

        return max;
    }

    template <typename T, bool C>
    uint64_t NDArray<T, C>::index_max() const {
        ConstNDIterator iter(*this);
        T value;
        T max = iter.next();
        uint64_t index_max = 0;

        for(uint64_t i = 1;i < this->m_size;i++) {
            if((value = iter.next()) > max) {
                max = value;
                index_max = iter.index();
            }
        }

        return index_max;
    }

    template <typename T, bool C>
    NDIndex NDArray<T, C>::ndindex_max() const {
        return this->unravel_index(this->index_max());
    }

    template <typename T, bool C>
    T NDArray<T, C>::min() const {
        ConstNDIterator iter(*this);
        T min = iter.next();

        for(uint64_t i = 1;i < this->m_size;i++) {
            min = common::min(min, iter.next());
        }

        return min;
    }

    template <typename T, bool C>
    uint64_t NDArray<T, C>::index_min() const {
        ConstNDIterator iter(*this);
        T value;
        T min = iter.next();
        uint64_t index_min = 0;

        for(uint64_t i = 1;i < this->m_size;i++) {
            if((value = iter.next()) < min) {
                min = value;
                index_min = iter.index();
            }
        }

        return index_min;
    }

    template <typename T, bool C>
    NDIndex NDArray<T, C>::ndindex_min() const {
        return this->unravel_index(this->index_min());
    }

    template <typename T, bool C>
    T& NDArray<T, C>::operator[](const NDIndex &ndindex) {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T, bool C>
    const T& NDArray<T, C>::operator[](const NDIndex &ndindex) const {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T, bool C>
    const NDArray<T, false> NDArray<T, C>::operator[](const SliceRanges &ranges) const {
        return NDArray<T, false>(*this, ranges);
    }

    template <typename T, bool C>
    NDArray<T, false> NDArray<T, C>::operator[](const SliceRanges &ranges) {
        return NDArray<T, false>(*this, ranges);
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator+=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() += value;
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator-=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() -= value;
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator*=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() *= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator/=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() /= value;
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    auto NDArray<T, C>::operator+(T2 value) const {

        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < ndarray.m_size;i++) {
            ndarray.m_data[i] = iter.next() + value;
        }

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    auto NDArray<T, C>::operator-(T2 value) const {

        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < ndarray.m_size;i++) {
            ndarray.m_data[i] = iter.next() - value;
        }

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    auto NDArray<T, C>::operator*(T2 value) const {

        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < ndarray.m_size;i++) {
            ndarray.m_data[i] = iter.next() * value;
        }

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, typename ENABLE>
    auto NDArray<T, C>::operator/(T2 value) const {

        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < ndarray.m_size;i++) {
            ndarray.m_data[i] = iter.next() / value;
        }

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator+(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array += ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator-(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array -= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator*(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array *= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator/(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array /= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator+=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() += rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator-=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() -= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator*=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() *= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator/=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() /= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator==(const NDArray<T2, C2> &ndarray) const {
        bool eq = this->m_shape == ndarray.m_shape;
        ConstNDIterator lhs_iter(*this);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < this->m_size && eq;i++) {
            eq = (lhs_iter.next() == rhs_iter.next());
        }

        return eq;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator!=(const NDArray<T2, C2> &ndarray) const {
        return !(*this == ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator>=(const NDArray<T2, C2> &ndarray) const {
        bool ge = this->m_shape == ndarray.m_shape;
        ConstNDIterator lhs_iter(*this);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < this->m_size && ge;i++) {
            ge = (lhs_iter.next() >= rhs_iter.next());
        }

        return ge;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator<=(const NDArray<T2, C2> &ndarray) const {
        bool le = this->m_shape == ndarray.m_shape;
        ConstNDIterator lhs_iter(*this);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < this->m_size && le;i++) {
            le = (lhs_iter.next() <= rhs_iter.next());
        }

        return le;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator>(const NDArray<T2, C2> &ndarray) const {
        return !(*this <= ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator<(const NDArray<T2, C2> &ndarray) const {
        return !(*this >= ndarray);
    }

    template <typename T, bool C> template <typename T2>
    NDArray<T, C>& NDArray<T, C>::operator^=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() ^= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2>
    NDArray<T, C>& NDArray<T, C>::operator&=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() &= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2>
    NDArray<T, C>& NDArray<T, C>::operator|=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() |= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2>
    NDArray<T, C>& NDArray<T, C>::operator<<=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() <<= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2>
    NDArray<T, C>& NDArray<T, C>::operator>>=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() >>= value;
        }
        
        return *this;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator^(T2 value) const {
        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = iter.next() ^ value;
        }
        
        return ndarray;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator&(T2 value) const {
        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = iter.next() & value;
        }
        
        return ndarray;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator|(T2 value) const {
        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = iter.next() | value;
        }
        
        return ndarray;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator<<(T2 value) const {
        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = iter.next() << value;
        }
        
        return ndarray;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator>>(T2 value) const {
        NDArray<types::combine_types_t<T, T2>> ndarray(
            new types::combine_types_t<T, T2>[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = iter.next() >> value;
        }
        
        return ndarray;
    }

    template <typename T, bool C>
    NDArray<T, C> NDArray<T, C>::operator~() const {
        NDArray<T, C> ndarray(new T[this->m_size], *this, true);
        ConstNDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            ndarray.m_data[i] = ~iter.next();
        }

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator^=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() ^= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator&=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() &= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator|=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() |= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator<<=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() <<= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator>>=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() >>= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator^(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array ^= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator&(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array &= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator|(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array |= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator<<(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array <<= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator>>(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array >>= ndarray;

        return output_array;
    }

    template <typename T, bool C> template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator%=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() %= value;
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator%=(T2 value) {
        NDIterator iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            iter.next() = (T)fmod((float64_t)iter.current(), (float64_t)value);
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2>
    auto NDArray<T, C>::operator%(T2 value) const {

        NDArray<types::combine_types_t<T, T2>> ndarray(*this);
        ndarray %= value;

        return ndarray;
    }

    template <typename T, bool C> template <typename T2, bool C2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int>>
    NDArray<T, C>& NDArray<T, C>::operator%=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() %= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return *this;
    }
    
    template <typename T, bool C> template <typename T2, bool C2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int>>
    NDArray<T, C>& NDArray<T, C>::operator%=(const NDArray<T2, C2> &ndarray) {
        NDArray<T, false> reorder = ndlib::broadcast_reorder(*this, ndarray);
        uint64_t size_ratio = this->m_size / ndarray.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(ndarray);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < ndarray.m_size;j++) {
                lhs_iter.next() += (T)fmod((float64_t)lhs_iter.current(), (float64_t)rhs_iter.next());
            }
            rhs_iter.reset();
        }

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    auto NDArray<T, C>::operator%(const NDArray<T2, C2> &ndarray) const {
        NDArray<types::combine_types_t<T, T2>> output_array(ndlib::broadcast(this->m_shape, ndarray.m_shape), 0);

        output_array += *this;
        output_array %= ndarray;

        return output_array;
    }

    template <typename T, bool C>
    void NDArray<T, C>::str_(std::string &str, uint8_t dim, uint64_t data_index, bool not_first, bool not_last) const {
        uint64_t dim_idx;
        uint64_t stride;

        if(not_first) {
            str += std::string(dim, ' ');
        }

        str.push_back('[');

        if(dim == this->m_ndim - 1) {
            stride = this->m_strides[dim];

            if(this->m_shape[dim]) {
                for(dim_idx = 0;dim_idx < this->m_shape[dim] - 1;dim_idx++) {
                    str += std::to_string(this->m_data[data_index]);
                    str.push_back(',');
                    str.push_back(' ');
                    data_index += stride;
                }

                str += std::to_string(this->m_data[data_index]);
            }

            str.push_back(']');
            if(not_last) {
                str.push_back('\n');
            }
            
            return;
        }

        if(this->m_shape[dim]) {
            this->str_(str, dim + 1, data_index, false, this->m_shape[dim] > 1);
            data_index += this->m_strides[dim];

            for(dim_idx = 1;dim_idx < this->m_shape[dim] - 1;dim_idx++) {
                this->str_(str, dim + 1, data_index, true, true);
                data_index += this->m_strides[dim];
            }
        }

        if(this->m_shape[dim] > 1) {
            this->str_(str, dim + 1, data_index, true, false);
        }
        
        str.push_back(']');
        
        if(!dim) {
            str.push_back('\n');
        }

        else if(not_last) {
            str += std::string(this->m_ndim - dim, '\n');
        }
    }
}