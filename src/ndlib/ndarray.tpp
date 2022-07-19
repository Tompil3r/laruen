
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

    template <typename T>
    NDArray<T>::~NDArray() {
        if(!this->m_base) {
            delete[] this->m_data;
        }
    }

    template <typename T>
    NDArray<T>::NDArray() noexcept : ArrayBase(), m_data(nullptr) {}

    template <typename T>
    NDArray<T>::NDArray(std::initializer_list<T> init_list) noexcept
    : ArrayBase(Shape{init_list.size()}, Strides{1}, Strides{init_list.size()}, init_list.size(), 1),
    m_data(new T[init_list.size()])
    {
        NDIter iter(this->m_data, *this);

        for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
            iter.next() = *list_ptr;
        }
    }

    template <typename T>
    NDArray<T>::NDArray(std::initializer_list<T> init_list, const Shape &shape) noexcept
    : NDArray<T>(shape)
    {
        assert(init_list.size() == this->m_size);

        NDIter iter(this->m_data, *this);

        for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
            iter.next() = *list_ptr;
        }
    }

    template <typename T>
    NDArray<T>::NDArray(T *data, const Shape &shape, const Strides &strides,
    const Strides &dim_sizes, uint_fast64_t size, uint_fast8_t ndim, const NDArray<T> *base) noexcept
    : ArrayBase(shape, strides, dim_sizes, size, ndim), m_data(data), m_base(base) {}

    template <typename T>
    NDArray<T>::NDArray(T *data, Shape &&shape, Strides &&strides, Strides &&dim_sizes,
    uint_fast64_t size, uint_fast8_t ndim, const NDArray<T> *base) noexcept
    : ArrayBase(std::move(shape), std::move(strides), std::move(dim_sizes), size, ndim),
    m_data(data), m_base(base) {}

    template <typename T>
    NDArray<T>::NDArray(const Shape &shape) noexcept
    : ArrayBase(shape), m_data(new T[this->m_size]) {}

    template <typename T>
    NDArray<T>::NDArray(const Shape &shape, T value) noexcept : NDArray<T>(shape) {
        this->fill(value);
    }

    template <typename T>
    NDArray<T>::NDArray(T *data, const ArrayBase &arraybase, const NDArray<T> *base) noexcept
    : ArrayBase(arraybase), m_data(data), m_base(base) {}

    template <typename T>
    NDArray<T>::NDArray(const NDArray<T> &ndarray) noexcept
    : NDArray<T>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T>
    NDArray<T>::NDArray(NDArray<T> &&ndarray) noexcept
    : ArrayBase(std::move(ndarray)), m_data(ndarray.m_data)
    {
        ndarray.m_data = nullptr;
    }

    template <typename T>
    NDArray<T>::NDArray(const Range<T> &range) noexcept
    : NDArray<T>(Shape{laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step)}) {
        T value = range.start;

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            this->m_data[i] = value;
            value += range.step;
        }
    }

    template <typename T>
    NDArray<T>::NDArray(const Range<T> &range, const Shape &shape)
    : NDArray<T>(shape)
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

    template <typename T>
    NDArray<T>::NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept
    : ArrayBase(axes.size(), axes.size() > 0)
    {
        uint_fast8_t axis;
        uint_fast64_t stride = 1;

        for(uint_fast8_t i = this->m_ndim;i-- > 0;) {
            axis = axes[i];
            this->m_shape[i] = arraybase.m_shape[axis];
            this->m_strides[i] = stride;
            stride *= this->m_shape[i];
            this->m_dim_sizes[i] = stride;
            this->m_size *= this->m_shape[i];
        }

        this->m_data = new T[this->m_size];
    }

    template <typename T>
    NDArray<T>::NDArray(NDArray<T> &ndarray, const SliceRanges &ranges) noexcept
    : NDArray<T>(ndarray.m_data, ndarray, ndarray.forward_base())
    {
        uint_fast8_t ndim = ranges.size();
        float64_t size_ratio = 1;

        for(uint_fast8_t dim = 0;dim < ndim;dim++) {
            size_ratio *= this->m_shape[dim];
            this->m_data += ranges[dim].start * this->m_strides[dim];
            this->m_strides[dim] = this->m_strides[dim] * ranges[dim].step;
            this->m_shape[dim] = laruen::ndlib::utils::ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
            this->m_dim_sizes[dim] = this->m_shape[dim] * this->m_strides[dim];
            size_ratio /= this->m_shape[dim];
        }

        this->m_size /= size_ratio;
    }

    template <typename T> template <typename T2>
    NDArray<T>::NDArray(const NDArray<T2> &ndarray) noexcept
    : NDArray<T>(new T[ndarray.m_size], ndarray)
    {
        this->copy_data_from(ndarray);
    }

    template <typename T> template <typename T2>
    NDArray<T>::NDArray(NDArray<T2> &&ndarray) noexcept
    : ArrayBase(std::move(ndarray)), m_data(new T[ndarray.m_size])
    {
        this->copy_data_from(ndarray);
    }

    template <typename T>
    NDArray<T>& NDArray<T>::operator=(const NDArray<T> &ndarray) noexcept {
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
        this->m_dim_sizes = ndarray.m_dim_sizes;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T>
    NDArray<T>& NDArray<T>::operator=(NDArray<T> &&ndarray) noexcept {
        if(this == &ndarray) {
            return *this;
        }

        if(!this->m_base) {
            delete[] this->m_data;
        }
        
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_dim_sizes = std::move(ndarray.m_dim_sizes);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = ndarray.m_free_mem; // check this
        
        this->m_data = ndarray.m_data;
        ndarray.m_data = nullptr;

        return *this;
    }

    template <typename T> template <typename T2>
    NDArray<T>& NDArray<T>::operator=(const NDArray<T2> &ndarray) noexcept {
        if(this->m_size != ndarray.m_size) {
            if(!this->m_base) {
                delete[] this->m_data;
            }
            this->m_data = new T[ndarray.m_size];
        }

        this->m_shape = ndarray.m_shape;
        this->m_strides = ndarray.m_strides;
        this->m_dim_sizes=  ndarray.m_dim_sizes;
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T> template <typename T2>
    NDArray<T>& NDArray<T>::operator=(NDArray<T2> &&ndarray) noexcept {
        this->m_data = new T[ndarray.m_size];
        this->m_shape = std::move(ndarray.m_shape);
        this->m_strides = std::move(ndarray.m_strides);
        this->m_dim_sizes = std::move(ndarray.m_dim_sizes);
        this->m_size = ndarray.m_size;
        this->m_ndim = ndarray.m_ndim;
        // this->m_free_mem = true; // check this

        this->copy_data_from(ndarray);

        return *this;
    }

    template <typename T> template <typename T2>
    void NDArray<T>::copy_data_from(const NDArray<T2> &ndarray) noexcept {
        NDIter to(this->m_data, *this);
        NDIter from(ndarray.m_data, ndarray);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            to.next() = from.next();
        }
    }

    template <typename T>
    void NDArray<T>::fill(T value) noexcept {
        NDIter iter(this->m_data, *this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            iter.next() = value;
        }
    }

    template <typename T> template <typename TR>
    NDArray<TR>& NDArray<T>::sum(const Axes &axes, NDArray<TR> &out) const noexcept {
        const NDArray<T> reorder = this->axes_reorder(axes);

        NDIter out_iter(out.m_data, out);
        NDIter this_iter(reorder.m_data, reorder);
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

    template <typename T>
    NDArray<T> NDArray<T>::sum(const Axes &axes) const noexcept {
        NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->sum(axes, out);
        return out;
    }
    
    template <typename T>
    T NDArray<T>::sum() const noexcept {
        T sum = 0;
        NDIter iter(this->m_data, *this);

        for(uint_fast64_t i = 0;i < this->m_size;i++) {
            sum += iter.next();
        }

        return sum;
    }

    template<typename T> template <typename TR>
    NDArray<TR>& NDArray<T>::max(const Axes &axes, NDArray<TR> &out) const noexcept {
        const NDArray<T> reorder = this->axes_reorder(axes);

        NDIter out_iter(out.m_data, out);
        NDIter this_iter(reorder.m_data, reorder);
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

    template <typename T>
    NDArray<T> NDArray<T>::max(const Axes &axes) const noexcept {
        NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->max(axes, out);
        return out;
    }

    template <typename T>
    T NDArray<T>::max() const noexcept {
        NDIter iter(this->m_data, *this);
        T max = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            max = laruen::math::common::max(max, iter.next());
        }

        return max;
    }

    template <typename T>
    NDArray<uint_fast64_t>& NDArray<T>::indices_max(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
        const NDArray<T> reorder = this->axes_reorder(axes);

        NDIter out_iter(out.m_data, out);
        NDIter src_iter(reorder.m_data, reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T max;
        T current;
        T *ptr_max;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            ptr_max = src_iter.ptr;
            max = src_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                current = src_iter.current();

                if(current > max) {
                    max = current;
                    ptr_max = src_iter.ptr;
                }
                src_iter.next();
            }
            out_iter.next() = ptr_max - this->m_data;
        }

        return out;
    }

    template <typename T>
    NDArray<uint_fast64_t> NDArray<T>::indices_max(const Axes &axes) const noexcept {
        NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->indices_max(axes, out);
        return out;
    }

    template <typename T>
    uint_fast64_t NDArray<T>::index_max() const noexcept {
        NDIter iter(this->m_data, *this);
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

    template <typename T>
    NDIndex NDArray<T>::ndindex_max() const noexcept {
        return this->unravel_index(this->index_max());
    }

    template <typename T> template <typename TR>
    NDArray<TR>& NDArray<T>::min(const Axes &axes, NDArray<TR> &out) const noexcept {
        const NDArray<T> reorder = this->axes_reorder(axes);

        NDIter out_iter(out.m_data, out);
        NDIter this_iter(reorder.m_data, reorder);
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

    template <typename T>
    NDArray<T> NDArray<T>::min(const Axes &axes) const noexcept {
        NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->min(axes, out);
        return out;
    }

    template <typename T>
    T NDArray<T>::min() const noexcept{
        NDIter iter(this->m_data, *this);
        T min = iter.next();

        for(uint_fast64_t i = 1;i < this->m_size;i++) {
            min = laruen::math::common::min(min, iter.next());
        }

        return min;
    }

    template <typename T>
    NDArray<uint_fast64_t>& NDArray<T>::indices_min(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
        const NDArray<T> reorder = this->axes_reorder(axes);

        NDIter out_iter(out.m_data, out);
        NDIter src_iter(reorder.m_data, reorder);
        uint_fast64_t sample_size = reorder.m_size / out.m_size;
        T min;
        T current;
        T *ptr_min;

        for(uint_fast64_t i = 0;i < out.m_size;i++) {
            ptr_min = src_iter.ptr;
            min = src_iter.next();
            
            for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                current = src_iter.current();

                if(current < min) {
                    min = current;
                    ptr_min = src_iter.ptr;
                }
                src_iter.next();
            }
            out_iter.next() = ptr_min - this->m_data;
        }

        return out;
    }

    template <typename T>
    NDArray<uint_fast64_t> NDArray<T>::indices_min(const Axes &axes) const noexcept {
        NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
        this->indices_min(axes, out);
        return out;
    }

    template <typename T>
    uint_fast64_t NDArray<T>::index_min() const noexcept {
        NDIter iter(this->m_data, *this);
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

    template <typename T>
    NDIndex NDArray<T>::ndindex_min() const noexcept {
        return this->unravel_index(this->index_min());
    }

    template <typename T>
    T& NDArray<T>::operator[](const NDIndex &ndindex) noexcept {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T>
    const T& NDArray<T>::operator[](const NDIndex &ndindex) const noexcept {
        return this->m_data[this->ravel_ndindex(ndindex)];
    }

    template <typename T>
    const NDArray<T> NDArray<T>::operator[](const SliceRanges &ranges) const noexcept {
        return NDArray<T>(*this, ranges);
    }

    template <typename T>
    NDArray<T> NDArray<T>::operator[](const SliceRanges &ranges) noexcept {
        return NDArray<T>(*this, ranges);
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator==(const NDArray<T2> &ndarray) const noexcept {
        bool eq = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(this->m_data, *this);
        NDIter rhs_iter(ndarray.m_data, ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && eq;i++) {
            eq = (lhs_iter.next() == rhs_iter.next());
        }

        return eq;
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator!=(const NDArray<T2> &ndarray) const noexcept {
        return !(*this == ndarray);
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator>=(const NDArray<T2> &ndarray) const noexcept {
        bool ge = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(this->m_data, *this);
        NDIter rhs_iter(ndarray.m_data, ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && ge;i++) {
            ge = (lhs_iter.next() >= rhs_iter.next());
        }

        return ge;
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator<=(const NDArray<T2> &ndarray) const noexcept {
        bool le = this->m_shape == ndarray.m_shape;
        NDIter lhs_iter(this->m_data, *this);
        NDIter rhs_iter(ndarray.m_data, ndarray);

        for(uint_fast64_t i = 0;i < this->m_size && le;i++) {
            le = (lhs_iter.next() <= rhs_iter.next());
        }

        return le;
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator>(const NDArray<T2> &ndarray) const noexcept {
        return !(*this <= ndarray);
    }

    template <typename T> template <typename T2>
    bool NDArray<T>::operator<(const NDArray<T2> &ndarray) const noexcept {
        return !(*this >= ndarray);
    }

    template <typename T>
    NDArray<T> NDArray<T>::transpose() noexcept {
        Shape t_shape(this->m_ndim);
        Strides t_strides(this->m_ndim);
        Strides t_dim_sizes(this->m_ndim);

        uint_fast8_t f = 0;
        uint_fast8_t b = this->m_ndim - 1;
        uint_fast8_t mid = this->m_ndim >> 1;

        t_shape[mid] = this->m_shape[mid];
        t_strides[mid] = this->m_strides[mid];
        t_dim_sizes[mid] = this->m_dim_sizes[mid];


        for(;f < mid;f++, b--) {
            t_shape[f] = this->m_shape[b];
            t_shape[b] = this->m_shape[f];
            t_strides[f] = this->m_strides[b];
            t_strides[b] = this->m_strides[f];
            t_dim_sizes[f] = this->m_dim_sizes[b];
            t_dim_sizes[b] = this->m_dim_sizes[f];
        }

        return NDArray<T>(this->m_data, std::move(t_shape),
        std::move(t_strides), std::move(t_dim_sizes), this->m_size, this->m_ndim, this->forward_base());
    }
    
    template <typename T>
    const NDArray<T> NDArray<T>::transpose() const noexcept {
        Shape t_shape(this->m_ndim);
        Strides t_strides(this->m_ndim);
        Strides t_dim_sizes(this->m_ndim);

        uint_fast8_t f = 0;
        uint_fast8_t b = this->m_ndim - 1;
        uint_fast8_t mid = this->m_ndim >> 1;

        t_shape[mid] = this->m_shape[mid];
        t_strides[mid] = this->m_strides[mid];
        t_dim_sizes[mid] = this->m_dim_sizes[mid];


        for(;f < mid;f++, b--) {
            t_shape[f] = this->m_shape[b];
            t_shape[b] = this->m_shape[f];
            t_strides[f] = this->m_strides[b];
            t_strides[b] = this->m_strides[f];
            t_dim_sizes[f] = this->m_dim_sizes[b];
            t_dim_sizes[b] = this->m_dim_sizes[f];
        }

        return NDArray<T>(this->m_data, std::move(t_shape),
        std::move(t_strides), std::move(t_dim_sizes), this->m_size, this->m_ndim, this->forward_base());
    }

    template <typename T>
    std::string NDArray<T>::str() const noexcept {
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

    template <typename T> template <typename T2>
    const NDArray<T> NDArray<T>::expand_to(const NDArray<T2> &expand_to) const noexcept {
        /* expand the dimensions of this to the dimensions of expand_to */
        
        NDArray<T> expansion(this->m_data, Shape(expand_to.m_shape), Strides(expand_to.m_ndim, 0),
        Strides(expand_to.m_ndim, 0), expand_to.m_size, expand_to.m_ndim, this->forward_base());
        
        uint_fast8_t to_idx = expand_to.m_ndim - this->m_ndim;

        for(uint_fast8_t this_idx = 0;this_idx < this->m_ndim;this_idx++, to_idx++) {
            if(expand_to.m_shape[to_idx] == this->m_shape[this_idx]) {
                expansion.m_strides[to_idx] = this->m_strides[this_idx];
                expansion.m_dim_sizes[to_idx] = this->m_dim_sizes[this_idx];
            }
        }

        return expansion;
    }

    template <typename T>
    const NDArray<T> NDArray<T>::axes_reorder(const Axes &axes) const noexcept {
        NDArray<T> reorder(this->m_data, Shape(this->m_ndim), Strides(this->m_ndim),
        Strides(this->m_ndim), this->m_size, this->m_ndim, this->forward_base());

        uint_fast8_t axes_added = 0;
        uint_fast8_t dest_idx = reorder.m_ndim - 1;
        
        for(uint_fast8_t i = axes.size();i-- > 0;) {
            reorder.m_shape[dest_idx] = this->m_shape[axes[i]];
            reorder.m_strides[dest_idx] = this->m_strides[axes[i]];
            reorder.m_dim_sizes[dest_idx] = this->m_dim_sizes[axes[i]];
            axes_added |= 1 << axes[i];
            dest_idx--;
        }

        for(uint_fast8_t i = reorder.m_ndim;i-- > 0;) {
            if(!(axes_added & (1 << i))) {
                reorder.m_shape[dest_idx] = this->m_shape[i];
                reorder.m_strides[dest_idx] = this->m_strides[i];
                reorder.m_dim_sizes[dest_idx] = this->m_dim_sizes[i];
                dest_idx--; 
            }
        }

        return reorder;
    }
}