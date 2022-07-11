
#include <cassert>
#include <ostream>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/ndlib/utils.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/range.h"
#include "src/ndlib/type_selection.h"
#include "src/ndlib/impl.h"
#include "src/math/common.h"

namespace laruen::ndlib {

    template <typename T, bool C>
    NDArray<T, C>::~NDArray() {
        if(!this->m_base) {
            delete[] this->m_data;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray() noexcept : ArrayBase(), m_data(nullptr) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(std::initializer_list<T> init_list) noexcept
    : ArrayBase(Shape{init_list.size()}, Strides{1}, init_list.size(), 1),
    m_data(new T[init_list.size()])
    {
        NDIter iter(*this);

        for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
            iter.next() = *list_ptr;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(std::initializer_list<T> init_list, const Shape &shape) noexcept
    : NDArray<T>(shape)
    {
        assert(init_list.size() == this->m_size);

        NDIter iter(*this);

        for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
            iter.next() = *list_ptr;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const Shape &shape, const Strides &strides,
    uint_fast64_t size, uint_fast8_t ndim, const NDArray<T, true> *base) noexcept
    : ArrayBase(shape, strides, size, ndim), m_data(data), m_base(base) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, Shape &&shape, Strides &&strides,
    uint_fast64_t size, uint_fast8_t ndim, const NDArray<T, true> *base) noexcept
    : ArrayBase(std::move(shape), std::move(strides), size, ndim), m_data(data), m_base(base) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape) noexcept
    : ArrayBase(shape), m_data(new T[this->m_size]) {}

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const Shape &shape, T value) noexcept : NDArray<T, C>(shape) {
        this->fill(value);
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(T *data, const ArrayBase &arraybase, const NDArray<T, true> *base) noexcept
    : ArrayBase(arraybase), m_data(data), m_base(base) {}

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
    : NDArray<T, C>(Shape{laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step)}) {
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
        if(laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step) != this->m_size) {
            throw std::invalid_argument("shape size does not match range");
        }

        T value = range.start;

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            this->m_data[i] = value;
            value += range.step;
        }
    }

    template <typename T, bool C>
    NDArray<T, C>::NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept
    : ArrayBase(axes.size(), axes.size() > 0)
    {
        uint_fast8_t axis;
        uint_fast64_t stride = 1;

        for(uint_fast8_t i = this->m_ndim;i-- > 0;) {
            axis = axes[i];
            this->m_shape[i] = arraybase.m_shape[axis];
            this->m_strides[i] = stride;
            this->m_size *= this->m_shape[i];
            stride *= this->m_shape[i];
        }

        this->m_data = new T[this->m_size];
    }

    template <typename T, bool C> template <bool C2>
    NDArray<T, C>::NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges) noexcept
    : NDArray<T, C>(ndarray.m_data, ndarray, ndarray.forward_base())
    {
        static_assert(!C, "cannot create sliced array with non-sliced type");

        uint_fast8_t ndim = ranges.size();
        float64_t size_ratio = 1;

        for(uint_fast8_t dim = 0;dim < ndim;dim++) {
            size_ratio *= this->m_shape[dim];
            this->m_data += ranges[dim].start * this->m_strides[dim];
            this->m_strides[dim] = this->m_strides[dim] * ranges[dim].step;
            this->m_shape[dim] = laruen::ndlib::utils::ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
            size_ratio /= this->m_shape[dim];
        }

        this->m_size /= size_ratio;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>::NDArray(const NDArray<T2, C2> &ndarray) noexcept
    : NDArray<T, C>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T, bool C> template <typename T2, bool C2>
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
            if(!this->m_base) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.m_strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T, C> &&ndarray) noexcept {
        if(this == &ndarray) {
            return *this;
        }

        if(!this->m_base) {
            delete[] this->m_data;
        }
        
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = ndarray.m_free_mem; // check this
        
        this->m_data = ndarray.m_data;
        ndarray.m_data = nullptr;

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator=(const NDArray<T2, C2> &ndarray) noexcept {
        if(this->m_size != ndarray.m_size) {
            if(!this->m_base) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.m_strides;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<T, C>& NDArray<T, C>::operator=(NDArray<T2, C2> &&ndarray) noexcept {
        this->m_data = new T[ndarray.m_size];
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

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

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::sum(const Axes &axes, NDArray<TR, CR> &out) const noexcept {
        const NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter this_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T sum;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            sum = 0;

            for(uint_fast64_t j = 0;j < sample_size;j++) {
                sum += this_iter.next();
            }
            out_iter.next() = sum;
        }

        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::sum(const Axes &axes) const noexcept {
        NDArray<T, true> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->sum(axes, out);
        return out;
    }
    
    template <typename T, bool C>
    T NDArray<T, C>::sum() const noexcept {
        T sum = 0;
        NDIter iter(*this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            sum += iter.next();
        }

        return sum;
    }

    template<typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::max(const Axes &axes, NDArray<TR, CR> &out) const noexcept {
        const NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter this_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T max;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            max = this_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                max = laruen::math::common::max(max, this_iter.next());
            }
            out_iter.next() = max;
        }

        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::max(const Axes &axes) const noexcept {
        NDArray<T, true> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->max(axes, out);
        return out;
    }

    template <typename T, bool C>
    T NDArray<T, C>::max() const noexcept {
        NDIter iter(*this);
        T max = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            max = laruen::math::common::max(max, iter.next());
        }

        return max;
    }

    template <typename T, bool C> template <bool CR>
    NDArray<uint_fast64_t, CR>& NDArray<T, C>::indices_max(const Axes &axes, NDArray<uint_fast64_t, CR> &out) const noexcept {
        const NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter src_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T max;
        T current;
        uint_fast64_t index_max;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            index_max = src_iter.index();
            max = src_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                current = src_iter.current();

                if(current > max) {
                    max = current;
                    index_max = src_iter.index();
                }
                src_iter.next();
            }
            out_iter.next() = index_max;
        }

        return out;
    }

    template <typename T, bool C>
    NDArray<uint_fast64_t, true> NDArray<T, C>::indices_max(const Axes &axes) const noexcept {
        NDArray<uint_fast64_t, true> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->indices_max(axes, out);
        return out;
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

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::min(const Axes &axes, NDArray<TR, CR> &out) const noexcept {
        const NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter this_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T min;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            min = this_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                min = laruen::math::common::min(min, this_iter.next());
            }
            out_iter.next() = min;
        }

        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::min(const Axes &axes) const noexcept {
        NDArray<T, true> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->min(axes, out);
        return out;
    }

    template <typename T, bool C>
    T NDArray<T, C>::min() const noexcept{
        NDIter iter(*this);
        T min = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            min = laruen::math::common::min(min, iter.next());
        }

        return min;
    }

    template <typename T, bool C> template <bool CR>
    NDArray<uint_fast64_t, CR>& NDArray<T, C>::indices_min(const Axes &axes, NDArray<uint_fast64_t, CR> &out) const noexcept {
        const NDArray<T, false> reorder = this->axes_reorder(axes);

        NDIter out_iter(out);
        NDIter src_iter(reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T min;
        T current;
        uint_fast64_t index_min;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            index_min = src_iter.index();
            min = src_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                current = src_iter.current();

                if(current < min) {
                    min = current;
                    index_min = src_iter.index();
                }
                src_iter.next();
            }
            out_iter.next() = index_min;
        }

        return out;
    }

    template <typename T, bool C>
    NDArray<uint_fast64_t, true> NDArray<T, C>::indices_min(const Axes &axes) const noexcept {
        NDArray<uint_fast64_t, true> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->indices_min(axes, out);
        return out;
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
        this->add(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() + rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::add_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->add_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::add_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->add_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::add(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->add_r(rhs, out) : this->add_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::add(const NDArray<T2, C2> &rhs) const {
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
        this->subtract(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() - rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::subtract_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->subtract_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::subtract_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->subtract_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::subtract(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->subtract_r(rhs, out) : this->subtract_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::subtract(const NDArray<T2, C2> &rhs) const {
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
        this->multiply(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() * rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::multiply_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->multiply_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::multiply_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->multiply_r(rhs);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::multiply(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->multiply_r(rhs, out) : this->multiply_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::multiply(const NDArray<T2, C2> &rhs) const {
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
        this->divide(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() / rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::divide_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->divide_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::divide_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->divide_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::divide(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->divide_r(rhs, out) : this->divide_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::divide(const NDArray<T2, C2> &rhs) const {
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
        this->bit_xor(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() ^ rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_xor_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_xor_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_xor_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_xor_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_xor(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_xor_r(rhs, out) : this->bit_xor_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_xor(const NDArray<T2, C2> &rhs) const {
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
        this->bit_and(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() & rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_and_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_and_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_and_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_and_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_and(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_and_r(rhs, out) : this->bit_and_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_and(const NDArray<T2, C2> &rhs) const {
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
        this->bit_or(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() | rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_or_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_or_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_or_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->bit_or_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::bit_or(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->bit_or_r(rhs, out) : this->bit_or_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::bit_or(const NDArray<T2, C2> &rhs) const {
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
        this->shl(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() << rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shl_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->shl_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shl_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->shl_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shl(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->shl_r(rhs, out) : this->shl_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shl(const NDArray<T2, C2> &rhs) const {
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
        this->shr(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = lhs_iter.next() >> rhs_iter.next();
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shr_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->shr_b(rhs, out);
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
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shr_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->shr_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::shr(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->shr_r(rhs, out) : this->shr_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::shr(const NDArray<T2, C2> &rhs) const {
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
        this->bit_not(out);
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
            lhs_iter.next() = laruen::math::common::remainder(lhs_iter.current(), rhs_iter.next());
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
            iter.next() = laruen::math::common::remainder(iter.current(), value);
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = laruen::math::common::remainder(this_iter.next(), value);
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::remainder(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        this->remainder(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::remainder_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->remainder_b(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::remainder_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->remainder_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::remainder(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->remainder_r(rhs, out) : this->remainder_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::remainder(const NDArray<T2, C2> &rhs) const {
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
            lhs_iter.next() = laruen::math::common::pow(lhs_iter.current(), rhs_iter.next());
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
            iter.next() = laruen::math::common::pow(iter.current(), value);
        }

        return *this;
	}

    template <typename T, bool C> template <typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power(T value, NDArray<TR, CR> &out) const noexcept {
        NDIter this_iter(*this);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = laruen::math::common::pow(this_iter.next(), value);
        }
        
        return out;
    }

    template <typename T, bool C>
    NDArray<T, true> NDArray<T, C>::power(T value) const noexcept {
        NDArray<T, true> out(this->m_shape);
        this->power(value, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDArray<T, false> lhs_exp = out.broadcast_expansion(*this);
        NDArray<T2, false> rhs_exp = out.broadcast_expansion(rhs);

        NDIter lhs_iter(lhs_exp);
        NDIter rhs_iter(rhs_exp);
        NDIter out_iter(out);

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            out_iter.next() = laruen::math::common::pow(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::power_b(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->power_b(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint64_t i = 0;i < this->m_size;i++) {
            out_iter.next() = laruen::math::common::pow(lhs_iter.next(), rhs_iter.next());
        }

        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::power_r(const NDArray<T2, C2> &rhs) const {
        NDArray<types::result_type_t<T, T2>, true> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape), 0);
        this->power_r(rhs, out);
        return out;
    }

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::power(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const {
        return this->m_shape == rhs.m_shape ? this->power_r(rhs, out) : this->power_b(rhs, out);
    }

    template <typename T, bool C> template <typename T2, bool C2>
    NDArray<types::result_type_t<T, T2>, true> NDArray<T, C>::power(const NDArray<T2, C2> &rhs) const {
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
    NDArray<T, false> NDArray<T, C>::transpose() noexcept {
        Shape t_shape(this->m_ndim);
        Strides t_strides(this->m_ndim);

        uint_fast8_t f = 0;
        uint_fast8_t b = this->m_ndim - 1;
        uint_fast8_t mid = this->m_ndim >> 1;

        t_shape[mid] = this->m_shape[mid];
        t_strides[mid] = this->m_strides[mid];

        for(;f < mid;f++, b--) {
            t_shape[f] = this->m_shape[b];
            t_shape[b] = this->m_shape[f];
            t_strides[f] = this->m_strides[b];
            t_strides[b] = this->m_strides[f];
        }

        return NDArray<T, false>(this->m_data, std::move(t_shape),
        std::move(t_strides), this->m_size, this->m_ndim, this->forward_base());
    }
    
    template <typename T, bool C>
    const NDArray<T, false> NDArray<T, C>::transpose() const noexcept {
        Shape t_shape(this->m_ndim);
        Strides t_strides(this->m_ndim);

        uint_fast8_t f = 0;
        uint_fast8_t b = this->m_ndim - 1;
        uint_fast8_t mid = this->m_ndim >> 1;

        t_shape[mid] = this->m_shape[mid];
        t_strides[mid] = this->m_strides[mid];

        for(;f < mid;f++, b--) {
            t_shape[f] = this->m_shape[b];
            t_shape[b] = this->m_shape[f];
            t_strides[f] = this->m_strides[b];
            t_strides[b] = this->m_strides[f];
        }

        return NDArray<T, false>(this->m_data, std::move(t_shape),
        std::move(t_strides), this->m_size, this->m_ndim, this->forward_base());
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
    const NDArray<T2, false> NDArray<T, C>::broadcast_expansion(const NDArray<T2, C2> &rhs) const noexcept {
        
        NDArray<T2, false> expansion(rhs.m_data, Shape(this->m_shape),
        Strides(this->m_ndim, 0), this->m_size, this->m_ndim, rhs.forward_base());
        
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
        this->m_size, this->m_ndim, C ? this : this->m_base);

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

    template <typename T, bool C> template <typename T2, bool C2, typename TR, bool CR>
    NDArray<TR, CR>& NDArray<T, C>::matmul_n3(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const noexcept {
        /* assumes the following:
            - shapes are valid
            - no broadcasting needed
            - min ndim == 2 (no dot product)
        */

        uint_fast8_t rows_axis = this->m_ndim - 2;
        uint_fast8_t cols_axis = this->m_ndim - 1;

        if(this->m_ndim <= 2) {
            impl::matmul_2d_n3(this->m_data, this->m_strides[rows_axis], this->m_strides[cols_axis],
            rhs.m_data, rhs.m_strides[rows_axis], rhs.m_strides[cols_axis], out.m_data, out.m_strides[rows_axis],
            out.m_strides[cols_axis], out.m_shape[rows_axis], out.m_shape[cols_axis], this->m_shape[cols_axis]);

            return out;
        }

        uint_fast64_t stacks = (this->m_size / this->m_shape[cols_axis]) / this->m_shape[rows_axis];
        uint_fast8_t iteration_axis = this->m_ndim - 3;
        NDIter lhs_iter(*this);
        NDIter rhs_iter(rhs);
        NDIter out_iter(out);

        for(uint_fast64_t stack = 0;stack < stacks;stack++) {
            impl::matmul_2d_n3(lhs_iter.ptr(), this->m_strides[rows_axis], this->m_strides[cols_axis],
            rhs_iter.ptr(), rhs.m_strides[rows_axis], rhs.m_strides[cols_axis], out_iter.ptr(), out.m_strides[rows_axis],
            out.m_strides[cols_axis], out.m_shape[rows_axis], out.m_shape[cols_axis], this->m_shape[cols_axis]);

            lhs_iter.next(iteration_axis);
            rhs_iter.next(iteration_axis);
            out_iter.next(iteration_axis);
        }

        return out;
    }
}