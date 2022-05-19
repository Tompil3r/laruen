
#include "src/ndlib/ndarray.h"
#include "src/ndlib/ndlib_types.h"
#include "src/ndlib/ndlib_utils.h"
#include "src/ndlib/nditer.h"
#include "src/utils/range.h"
#include "src/utils/strings.h"
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
using laruen::utils::Range;

namespace laruen::ndlib {

    template <typename T, bool C>
    NDArray<T, C>::~NDArray() {
        if(this->m_free_mem) {
            delete[] this->m_data;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray() noexcept : ArrayBase(), m_data(nullptr) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const Shape &shape, const Strides &strides,
    uint_fast64_t size, uint_fast8_t ndim, bool free_mem) noexcept
    : ArrayBase(shape, strides, size, ndim, free_mem), m_data(data) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, Shape &&shape, Strides &&strides,
    uint_fast64_t size, uint_fast8_t ndim, bool free_mem) noexcept
    : ArrayBase(std::move(shape), std::move(strides), size, ndim, free_mem), m_data(data) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape) noexcept
    : ArrayBase(shape), m_data(new T[this->m_size]) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape, T value) noexcept : NDArray<T, C>(shape) {
        this->fill(value);
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const ArrayBase &base) noexcept
    : ArrayBase(base), m_data(data) {}
    
    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const ArrayBase &base, bool free_mem) noexcept
    : ArrayBase(base, free_mem), m_data(data) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const NDArray<T, C> &ndarray) noexcept
    : NDArray<T, C>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(NDArray<T, C> &&ndarray) noexcept
    : ArrayBase(std::move(ndarray)), m_data(ndarray.m_data)
    {
        ndarray.m_data = nullptr;
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Range<T> &range) noexcept
    : NDArray<T, C>(Shape{ceil_index((range.end - range.start) / range.step)}) {
        T value = range.start;

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            this->m_data[i] = value;
            value += range.step;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Range<T> &range, const Shape &shape)
    : NDArray<T, C>(shape)
    {
        if(ceil_index((range.end - range.start) / range.step) != this->m_size) {
            throw std::invalid_argument("shape size does not match range");
        }

        T value = range.start;

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            this->m_data[i] = value;
            value += range.step;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const ArrayBase &base, const Axes &axes) noexcept
    : ArrayBase(axes.size(), true, axes.size() > 0)
    {
        uint_fast8_t axis;
        uint_fast64_t stride = 1;

        for(uint_fast8_t i = this->m_ndim;i-- > 0;) {
            axis = axes[i];
            this->m_shape[i] = base.m_shape[axis];
            this->m_strides[i] = stride;
            this->m_size *= this->m_shape[i];
            stride *= this->m_shape[i];
        }

        this->m_data = new T[this->m_size];
    }

    template <typename T, bool C> template <bool C2>
    NDArray<T, C>::NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges) noexcept
    : NDArray<T, C>(ndarray.m_data, ndarray, false)
    {
        static_assert(!C, "cannot create sliced array with non-sliced type");

        uint_fast8_t ndim = ranges.size();
        float64_t size_ratio = 1;

        for(uint_fast8_t dim = 0;dim < ndim;dim++) {
            size_ratio *= this->m_shape[dim];
            this->m_data += ranges[dim].start * this->m_strides[dim];
            this->m_strides[dim] = this->m_strides[dim] * ranges[dim].step;
            this->m_shape[dim] = ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
            size_ratio /= this->m_shape[dim];
        }

        this->m_size /= size_ratio;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>::NDArray(const NDArray<T2, C2> &ndarray) noexcept
    : NDArray<T, C>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>::NDArray(NDArray<T2, C2> &&ndarray) noexcept
    : ArrayBase(std::move(ndarray)), m_data(new T[ndarray.m_size])
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::operator=(const NDArray<T, C> &ndarray) noexcept {
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
        this->m_strides = ndarray.m_strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T, C> &&ndarray) noexcept {
        if(this == &ndarray) {
            return *this;
        }

        if(this->m_free_mem) {
            delete[] this->m_data;
        }
        
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = ndarray.m_free_mem;
        
        this->m_data = ndarray.m_data;
        ndarray.m_data = nullptr;

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator=(const NDArray<T2, C2> &ndarray) noexcept {
        if(this->m_size != ndarray.m_size) {
            if(this->m_free_mem) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.m_strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename ENABLE>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T2, C2> &&ndarray) noexcept {
        this->m_data = new T[ndarray.m_size];
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        this->m_free_mem = true;

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    void NDArray<T, C>::copy_data_from(const NDArray<T2, C2> &ndarray) noexcept {
        NDIter to(*this);
        NDIter from(ndarray);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            to.next() = from.next();
        }
    }

    template <typename T, bool C>
    void NDArray<T, C>::fill(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() = value;
        }
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::max(const Axes &axes) const noexcept {
        NDArray<T, true> out(*this, ndlib::utils::remaining_axes(axes, this->m_ndim));
        NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter this_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T max;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            max = this_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                max = math::common::max(max, this_iter.next());
            }
            out_iter.next() = max;
        }

        return out;
    }

    template <typename T, bool C>
    T NDArray<T, C>::max() const noexcept {
        NDIter iter(*this);
        T max = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            max = common::max(max, iter.next());
        }

        return max;
    }

    template <typename T, bool C>
    uint_fast64_t NDArray<T, C>::index_max() const noexcept {
        NDIter iter(*this);
        T value;
        T max = iter.next();
        uint_fast64_t index_max = 0;

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            if((value = iter.next()) > max) {
                max = value;
                index_max = iter.index();
            }
        }

        return index_max;
    }

    template <typename T, bool C>
    NDIndex NDArray<T, C>::ndindex_max() const noexcept {
        return this->unravel_index(this->index_max());
    }

    template <typename T, bool C>
    T NDArray<T, C>::min() const noexcept{
        NDIter iter(*this);
        T min = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            min = common::min(min, iter.next());
        }

        return min;
    }

    template <typename T, bool C>
    uint_fast64_t NDArray<T, C>::index_min() const noexcept {
        NDIter iter(*this);
        T value;
        T min = iter.next();
        uint_fast64_t index_min = 0;

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            if((value = iter.next()) < min) {
                min = value;
                index_min = iter.index();
            }
        }

        return index_min;
    }

    template <typename T, bool C>
    NDIndex NDArray<T, C>::ndindex_min() const noexcept {
        return this->unravel_index(this->index_min());
    }

    template <typename T, bool C>
    T& NDArray<T, C>::operator[](const NDIndex &ndindex) noexcept {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T, bool C>
    const T& NDArray<T, C>::operator[](const NDIndex &ndindex) const noexcept {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T, bool C>
    const NDArray<T, false> NDArray<T, C>::operator[](const SliceRanges &ranges) const noexcept {
        return NDArray<T, false>(*this, ranges);
    }

    template <typename T, bool C>
    NDArray<T, false> NDArray<T, C>::operator[](const SliceRanges &ranges) noexcept {
        return NDArray<T, false>(*this, ranges);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::add_eq_b(const NDArray<T2, C2> &rhs) {
        return this->add_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::add_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() += rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::add_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->add_eq_r(rhs) : this->add_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::add_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() += value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() + value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::add(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() + value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.add_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::add_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.add_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() + rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::add_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() + rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->add_r(rhs, out) : this->add_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::add(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->add_r(rhs) : this->add_b(rhs);
    }


    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::subtract_eq_b(const NDArray<T2, C2> &rhs) {
        return this->subtract_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::subtract_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() -= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::subtract_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->subtract_eq_r(rhs) : this->subtract_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::subtract_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() -= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() - value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::subtract(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() - value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.subtract_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::subtract_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.subtract_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() - rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::subtract_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() - rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->subtract_r(rhs, out) : this->subtract_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::subtract(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->subtract_r(rhs) : this->subtract_b(rhs);
    }
    
    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::multiply_eq_b(const NDArray<T2, C2> &rhs) {
        return this->multiply_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::multiply_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() *= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::multiply_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->multiply_eq_r(rhs) : this->multiply_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::multiply_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() *= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() * value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::multiply(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() * value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.multiply_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::multiply_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.multiply_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() * rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::multiply_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() * rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->multiply_r(rhs, out) : this->multiply_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::multiply(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->multiply_r(rhs) : this->multiply_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::divide_eq_b(const NDArray<T2, C2> &rhs) {
        return this->divide_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::divide_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() /= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::divide_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->divide_eq_r(rhs) : this->divide_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::divide_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() /= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() / value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::divide(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() / value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.divide_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::divide_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.divide_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() / rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::divide_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() / rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->divide_r(rhs, out) : this->divide_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::divide(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->divide_r(rhs) : this->divide_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_xor_eq_b(const NDArray<T2, C2> &rhs) {
        return this->bit_xor_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_xor_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() ^= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_xor_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->bit_xor_eq_r(rhs) : this->bit_xor_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::bit_xor_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() ^= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() ^ value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::bit_xor(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() ^ value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.bit_xor_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_xor_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.bit_xor_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() ^ rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_xor_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() ^ rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_xor_r(rhs, out) : this->bit_xor_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_xor(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->bit_xor_r(rhs) : this->bit_xor_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_and_eq_b(const NDArray<T2, C2> &rhs) {
        return this->bit_and_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_and_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() &= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_and_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->bit_and_eq_r(rhs) : this->bit_and_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::bit_and_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() &= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() & value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::bit_and(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() & value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.bit_and_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_and_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.bit_and_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() & rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_and_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() & rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_and_r(rhs, out) : this->bit_and_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_and(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->bit_and_r(rhs) : this->bit_and_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_or_eq_b(const NDArray<T2, C2> &rhs) {
        return this->bit_or_eq_r(this->broadcast_expansion(rhs));
	}
    
    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_or_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() |= rhs_iter.next();
        }

        return *this;
	}
    
    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::bit_or_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->bit_or_eq_r(rhs) : this->bit_or_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::bit_or_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() |= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() | value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::bit_or(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() | value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.bit_or_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_or_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.bit_or_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() | rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_or_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() | rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_or_r(rhs, out) : this->bit_or_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::bit_or(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->bit_or_r(rhs) : this->bit_or_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shl_eq_b(const NDArray<T2, C2> &rhs) {
        return this->shl_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shl_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() <<= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shl_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->shl_eq_r(rhs) : this->shl_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::shl_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() <<= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() << value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::shl(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() << value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.shl_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shl_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.shl_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() << rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shl_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() << rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->shl_r(rhs, out) : this->shl_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shl(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->shl_r(rhs) : this->shl_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shr_eq_b(const NDArray<T2, C2> &rhs) {
        return this->shr_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shr_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() >>= rhs_iter.next();
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::shr_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->shr_eq_r(rhs) : this->shr_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::shr_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() >>= value;
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = this_iter.next() >> value;
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::shr(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = iter.next() >> value;
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.shr_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shr_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.shr_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = lhs_iter.next() >> rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shr_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = lhs_iter.next() >> rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->shr_r(rhs, out) : this->shr_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::shr(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->shr_r(rhs) : this->shr_b(rhs);
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::bit_not_eq() noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() = ~iter.current();
        }

        return *this;
    }

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_not(NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = ~this_iter.next();
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::bit_not() const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = ~iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::remainder_eq_b(const NDArray<T2, C2> &rhs) {
        return this->remainder_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::remainder_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() = math::common::remainder(lhs_iter.current(), rhs_iter.next());
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::remainder_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->remainder_eq_r(rhs) : this->remainder_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::remainder_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() = math::common::remainder(iter.current(), value);
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = math::common::remainder(this_iter.next(), value);
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::remainder(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = math::common::remainder(iter.next(), value);
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.remainder_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::remainder_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.remainder_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = math::common::remainder(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::remainder_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = math::common::remainder(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->remainder_r(rhs, out) : this->remainder_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::remainder(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->remainder_r(rhs) : this->remainder_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::power_eq_b(const NDArray<T2, C2> &rhs) {
        return this->power_eq_r(this->broadcast_expansion(rhs));
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::power_eq_r(const NDArray<T2, C2> &rhs) {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            lhs_iter.next() = math::common::pow(lhs_iter.current(), rhs_iter.next());
        }

        return *this;
	}

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::power_eq(const NDArray<T2, C2> &rhs) {
        return this->m_shape == rhs.m_shape ? this->power_eq_r(rhs) : this->power_eq_b(rhs);
	}

	template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::power_eq(T value) noexcept {
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() = math::common::pow(iter.current(), value);
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = math::common::pow(this_iter.next(), value);
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::power(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        NDIter iter(*this);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = math::common::pow(iter.next() - value);
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        out.fill(0);
        out.add_eq_b(*this);
        out.power_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::power_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        out.add_eq_b(*this);
        out.power_eq_b(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = math::common::pow(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::power_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::combine_types_t<T, T2>, true> out(ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out.m_data[i] = math::common::pow(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->power_r(rhs, out) : this->power_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::combine_types_t<T, T2>, true> NDArray<T, C>::power(const NDArray<T2, C2> &rhs) const {
        return this->m_shape == rhs.m_shape ? this->power_r(rhs) : this->power_b(rhs);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator==(const NDArray<T2, C2> &ndarray) const noexcept {
        bool eq = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(*this);
        NDIter rhs_iter(ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && eq;i++) {
            eq = (lhs_iter.next() == rhs_iter.next());
        }

        return eq;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator!=(const NDArray<T2, C2> &ndarray) const noexcept {
        return !(*this == ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator>=(const NDArray<T2, C2> &ndarray) const noexcept {
        bool ge = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(*this);
        NDIter rhs_iter(ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && ge;i++) {
            ge = (lhs_iter.next() >= rhs_iter.next());
        }

        return ge;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator<=(const NDArray<T2, C2> &ndarray) const noexcept {
        bool le = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(*this);
        NDIter rhs_iter(ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && le;i++) {
            le = (lhs_iter.next() <= rhs_iter.next());
        }

        return le;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator>(const NDArray<T2, C2> &ndarray) const noexcept {
        return !(*this <= ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    bool NDArray<T, C>::operator<(const NDArray<T2, C2> &ndarray) const noexcept {
        return !(*this >= ndarray);
    }

    template <typename T, bool C>
    std::string NDArray<T, C>::str() const noexcept {
        NDIndex ndindex(this->m_ndim, 0);
        uint_fast64_t index = 0;
        uint_fast8_t dim = 0;
        std::string str;

        if(!this->m_size) {
            str.push_back('[');
            str.push_back(']');
            return str;
        }

        str.reserve(this->m_size * (this->m_ndim / 2) * 19);

        for(uint_fast64_t i = 0;i < this->m_size - 1;i++) {
            if(!ndindex[this->m_ndim - 1]) {
                str += std::string(dim, ' ') + std::string(this->m_ndim - dim, '[');
            }

            str += std::to_string(this->m_data[index]);
            ndindex[this->m_ndim - 1]++;
            index += this->m_strides[this->m_ndim - 1];

            for(dim = this->m_ndim;dim-- > 1 && ndindex[dim] >= this->m_shape[dim];) {
                ndindex[dim] = 0;
                ndindex[dim - 1]++;
                index += this->m_strides[dim - 1] - m_shape[dim] * this->m_strides[dim];
                str.push_back(']');
            }
            dim++;

            if(dim == this->m_ndim) {
                str.push_back(',');
                str.push_back(' ');
            }

            str += std::string(this->m_ndim - dim, '\n');
        }

        if(!ndindex[this->m_ndim - 1]) {
                str += std::string(dim, ' ') + std::string(this->m_ndim - dim, '[');
        }

        str += std::to_string(this->m_data[index]);
        str += std::string(this->m_ndim, ']');

        return str;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    const NDArray<T2, false> NDArray<T, C>::broadcast_expansion(const NDArray<T2, C2> &rhs) noexcept {
        
        NDArray<T2, false> expansion(rhs.m_data, Shape(this->m_shape),
        Strides(this->m_ndim, 0), this->m_size, this->m_ndim, false);
        
        uint_fast8_t lidx = this->m_ndim - rhs.m_ndim;

        for(uint_fast8_t ridx = 0;ridx < rhs.m_ndim;ridx++) {
            if(this->m_shape[lidx] == rhs.m_shape[ridx]) {
                expansion.m_strides[lidx] = rhs.m_strides[ridx];
            }
            lidx++;
        }

        return expansion;
    }

    template <typename T, bool C>
    const NDArray<T, false> NDArray<T, C>::axes_reorder(const Axes &axes) const noexcept {
        NDArray<T, false> reorder(this->m_data, Shape(this->m_ndim), Strides(this->m_ndim),
        this->m_size, this->m_ndim, false);

        uint_fast8_t axes_added = 0;
        uint_fast8_t dest_idx = reorder.m_ndim - 1;
        
        for(uint_fast8_t i = axes.size();i-- > 0;) {
            reorder.m_shape[dest_idx] = this->m_shape[axes[i]];
            reorder.m_strides[dest_idx] = this->m_strides[axes[i]];
            axes_added |= 1 << axes[i];
            dest_idx--;
        }

        for(uint_fast8_t i = reorder.m_ndim;i-- > 0;) {
            if(!(axes_added & (1 << i))) {
                reorder.m_shape[dest_idx] = this->m_shape[i];
                reorder.m_strides[dest_idx] = this->m_strides[i];
                dest_idx--; 
            }
        }

        return reorder;
    }
}