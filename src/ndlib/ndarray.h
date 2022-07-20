
#ifndef NDLIB_NDARRAY_H_
#define NDLIB_NDARRAY_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <type_traits>
#include <ostream>
#include <initializer_list>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "src/ndlib/utils.h"
#include "src/ndlib/types.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/array_base.h"
#include "src/ndlib/range.h"
#include "src/ndlib/type_selection.h"
#include "src/ndlib/impl.h"
#include "src/math/common.h"
#include "src/math/bits.h"


namespace laruen::ndlib {

    template <typename T = float64_t>
    class NDArray : public ArrayBase {

        template <typename> friend class NDArray;
        friend struct Impl;

        private:
            T *m_data;
            const NDArray<T> *m_base = nullptr;

        public:
            typedef T DType;

            ~NDArray() {
                if(!this->m_base) {
                    delete[] this->m_data;
                }
            }

            // constructors and assignment operators
            NDArray() noexcept : ArrayBase(), m_data(nullptr)
            {}

            NDArray(std::initializer_list<T> init_list) noexcept
            : ArrayBase(Shape{init_list.size()}, Strides{1}, Strides{init_list.size()}, init_list.size(), 1),
            m_data(new T[init_list.size()])
            {
                NDIter iter(this->m_data, *this);

                for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
                    iter.next() = *list_ptr;
                }
            }

            NDArray(std::initializer_list<T> init_list, const Shape &shape) noexcept
            : NDArray(shape)
            {
                assert(init_list.size() == this->m_size);

                NDIter iter(this->m_data, *this);

                for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
                    iter.next() = *list_ptr;
                }
            }

            NDArray(T *data, const Shape &shape, const Strides &strides, const Strides &dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, const NDArray *base = nullptr) noexcept
            : ArrayBase(shape, strides, dim_sizes, size, ndim), m_data(data), m_base(base)
            {}

            NDArray(T *data, Shape &&shape, Strides &&strides, Strides &&dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, const NDArray *base = nullptr) noexcept
            : ArrayBase(std::move(shape), std::move(strides), std::move(dim_sizes), size, ndim),
            m_data(data), m_base(base)
            {}
            
            explicit NDArray(const Shape &shape) noexcept
            : ArrayBase(shape), m_data(new T[this->m_size])
            {}
            
            NDArray(const Shape &shape, T value) noexcept
            : NDArray(shape) {
                this->fill(value);
            }
            
            NDArray(T *data, const ArrayBase &arraybase, const NDArray<T> *base = nullptr) noexcept
            : ArrayBase(arraybase), m_data(data), m_base(base)
            {}
            
            NDArray(const NDArray &ndarray) noexcept 
            : NDArray(new T[ndarray.m_size], ndarray) {
                this->copy_data_from(ndarray);
            }
            
            NDArray(NDArray &&ndarray) noexcept
            : ArrayBase(std::move(ndarray)), m_data(ndarray.m_data) {
                ndarray.m_data = nullptr;
            }
            
            explicit NDArray(const Range<T> &range) noexcept
            : NDArray(Shape{laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step)})
            {
                T value = range.start;

                for(uint_fast64_t i = 0;i < this->m_size;i++) {
                    this->m_data[i] = value;
                    value += range.step;
                }
            }
            
            NDArray(const Range<T> &range, const Shape &shape)
            : NDArray(shape)
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
            
            NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept
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
            
            NDArray(NDArray<T> &ndarray, const SliceRanges &ranges) noexcept
            : NDArray(ndarray.m_data, ndarray, ndarray.forward_base())
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
            
            template <typename T2>
            NDArray(const NDArray<T2> &ndarray) noexcept
            : NDArray<T>(new T[ndarray.m_size], ndarray) {
                this->copy_data_from(ndarray);
            }
            
            template <typename T2>
            NDArray(NDArray<T2> &&ndarray) noexcept
            : ArrayBase(std::move(ndarray)), m_data(new T[ndarray.m_size]) {
                this->copy_data_from(ndarray);
            }

            NDArray& operator=(const NDArray &ndarray) noexcept {
                if(this == &ndarray) {
                    return *this;
                }

                if(this->m_size != ndarray.m_size) {
                    if(!this->m_base) {
                        delete[] this->m_data;
                    }
                    this->m_data = new T[ndarray.m_size];
                    // base update is required - 
                    // creating new data means
                    // 'this' object is the owner
                    this->m_base = nullptr;
                }

                this->m_shape = ndarray.m_shape;
                this->m_strides = ndarray.m_strides;
                this->m_dim_sizes = ndarray.m_dim_sizes;
                this->m_size = ndarray.m_size;
                this->m_ndim = ndarray.m_ndim;

                this->copy_data_from(ndarray);

                return *this;
            }
            
            NDArray& operator=(NDArray &&ndarray) noexcept {
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
                // base update required since memory space
                // of data is not ours
                this->m_base = ndarray.m_base;
                
                this->m_data = ndarray.m_data;
                ndarray.m_data = nullptr;

                return *this;
            }
            
            template <typename T2>
            NDArray& operator=(const NDArray<T2> &ndarray) noexcept {
                if(this->m_size != ndarray.m_size) {
                    if(!this->m_base) {
                        delete[] this->m_data;
                    }
                    this->m_data = new T[ndarray.m_size];
                    // base update is required - 
                    // creating new data means
                    // 'this' object is the owner
                    this->m_base = nullptr;
                }

                this->m_shape = ndarray.m_shape;
                this->m_strides = ndarray.m_strides;
                this->m_dim_sizes=  ndarray.m_dim_sizes;
                this->m_size = ndarray.m_size;
                this->m_ndim = ndarray.m_ndim;

                this->copy_data_from(ndarray);

                return *this;
            }
            
            template <typename T2>
            NDArray& operator=(NDArray<T2> &&ndarray) noexcept {
                this->m_data = new T[ndarray.m_size];
                this->m_shape = std::move(ndarray.m_shape);
                this->m_strides = std::move(ndarray.m_strides);
                this->m_dim_sizes = std::move(ndarray.m_dim_sizes);
                this->m_size = ndarray.m_size;
                this->m_ndim = ndarray.m_ndim;
                // base change needs to be done -
                // new data means 'this' object
                // is the owner of the data
                this->m_base = nullptr;

                this->copy_data_from(ndarray);

                return *this;
            }

            // utility functions
            template <typename T2>
            void copy_data_from(const NDArray<T2> &ndarray) noexcept {
                NDIter to(this->m_data, *this);
                NDIter from(ndarray.m_data, ndarray);

                for(uint_fast64_t i = 0;i < this->m_size;i++) {
                    to.next() = from.next();
                }
            }
            
            void fill(T value) noexcept {
                NDIter iter(this->m_data, *this);

                for(uint_fast64_t i = 0;i < this->m_size;i++) {
                    iter.next() = value;
                }
            }

            // computational functions on the array
            template <typename TR>
            NDArray<TR>& sum(const Axes &axes, NDArray<TR> &out) const noexcept {
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

            NDArray<T> sum(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
                this->sum(axes, out);
                return out;
            }

            T sum() const noexcept {
                T sum = 0;
                NDIter iter(this->m_data, *this);

                for(uint_fast64_t i = 0;i < this->m_size;i++) {
                    sum += iter.next();
                }

                return sum;
            }

            template <typename TR>
            NDArray<TR>& max(const Axes &axes, NDArray<TR> &out) const noexcept {
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

            NDArray<T> max(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
                this->max(axes, out);
                return out;
            }
            
            T max() const noexcept {
                NDIter iter(this->m_data, *this);
                T max = iter.next();

                for(uint_fast64_t i = 1;i < this->m_size;i++) {
                    max = laruen::math::common::max(max, iter.next());
                }

                return max;
            }
            
            NDArray<uint_fast64_t>& indices_max(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
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

            NDArray<uint_fast64_t> indices_max(const Axes &axes) const noexcept {
                NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
                this->indices_max(axes, out);
                return out;
            }

            uint_fast64_t index_max() const noexcept {
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
            
            NDIndex ndindex_max() const noexcept {
                return this->unravel_index(this->index_max());
            }

            template <typename TR>
            NDArray<TR>& min(const Axes &axes, NDArray<TR> &out) const noexcept {
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
            
            NDArray<T> min(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
                this->min(axes, out);
                return out;
            }

            T min() const noexcept {
                NDIter iter(this->m_data, *this);
                T min = iter.next();

                for(uint_fast64_t i = 1;i < this->m_size;i++) {
                    min = laruen::math::common::min(min, iter.next());
                }

                return min;
            }
            
            NDArray<uint_fast64_t>& indices_min(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
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

            NDArray<uint_fast64_t> indices_min(const Axes &axes) const noexcept {
                NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->m_ndim));
                this->indices_min(axes, out);
                return out;
            }

            uint_fast64_t index_min() const noexcept {
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
            
            NDIndex ndindex_min() const noexcept {
                return this->unravel_index(this->index_min());
            }

            // indexing and slicing operators
            T& operator[](const NDIndex &ndindex) noexcept {
                return this->m_data[this->ravel_ndindex(ndindex)];
            }

            const T& operator[](const NDIndex &ndindex) const noexcept {
                return this->m_data[this->ravel_ndindex(ndindex)];
            }

            NDArray<T> operator[](const SliceRanges &ranges) noexcept {
                return NDArray<T>(*this, ranges);
            }
            
            const NDArray<T> operator[](const SliceRanges &ranges) const noexcept {
                return NDArray<T>(*this, ranges);
            }

            // bool operators between arrays
            template <typename T2>
            bool operator==(const NDArray<T2> &ndarray) const noexcept {
                bool eq = this->m_shape == ndarray.m_shape;
                NDIter lhs_iter(this->m_data, *this);
                NDIter rhs_iter(ndarray.m_data, ndarray);

                for(uint_fast64_t i = 0;i < this->m_size && eq;i++) {
                    eq = (lhs_iter.next() == rhs_iter.next());
                }

                return eq;
            }
            
            template <typename T2>
            bool operator!=(const NDArray<T2> &ndarray) const noexcept {
                return !(*this == ndarray);
            }
            
            template <typename T2>
            bool operator>=(const NDArray<T2> &ndarray) const noexcept {
                bool ge = this->m_shape == ndarray.m_shape;
                NDIter lhs_iter(this->m_data, *this);
                NDIter rhs_iter(ndarray.m_data, ndarray);

                for(uint_fast64_t i = 0;i < this->m_size && ge;i++) {
                    ge = (lhs_iter.next() >= rhs_iter.next());
                }

                return ge;
            }
            
            template <typename T2>
            bool operator<=(const NDArray<T2> &ndarray) const noexcept {
                bool le = this->m_shape == ndarray.m_shape;
                NDIter lhs_iter(this->m_data, *this);
                NDIter rhs_iter(ndarray.m_data, ndarray);

                for(uint_fast64_t i = 0;i < this->m_size && le;i++) {
                    le = (lhs_iter.next() <= rhs_iter.next());
                }

                return le;
            }
            
            template <typename T2>
            bool operator>(const NDArray<T2> &ndarray) const noexcept {
                return !(*this <= ndarray);
            }
            
            template <typename T2>
            bool operator<(const NDArray<T2> &ndarray) const noexcept {
                return !(*this >= ndarray);
            }

            NDArray<T> transpose() noexcept {
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

            const NDArray<T> transpose() const noexcept {
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

            // string function
            std::string str() const noexcept {
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

        private:
            // private utility functions
            template <typename T2>
            const NDArray<T> expansion(const NDArray<T2> &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expansion */
                
                NDArray<T> expansion(this->m_data, Shape(expand_to.m_shape), Strides(expand_to.m_ndim, 0),
                Strides(expand_to.m_ndim, 0), expand_to.m_size, expand_to.m_ndim, this->forward_base());
                
                uint_fast8_t expansion_idx = expansion.m_ndim - this->m_ndim;

                for(uint_fast8_t expanded_idx = 0;expanded_idx < this->m_ndim;expanded_idx++, expansion_idx++) {
                    if(expansion.m_shape[expansion_idx] == this->m_shape[expanded_idx]) {
                        expansion.m_strides[expansion_idx] = this->m_strides[expanded_idx];
                        expansion.m_dim_sizes[expansion_idx] = this->m_dim_sizes[expanded_idx];
                    }
                }

                return expansion;
            }

            template <typename T2>
            const NDArray<T> matmul_expansion(const NDArray<T2> &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expand_to */
                
                NDArray<T> expansion(this->m_data, Shape(expand_to.m_shape), Strides(expand_to.m_ndim, 0),
                Strides(expand_to.m_ndim, 0), expand_to.m_size, expand_to.m_ndim, this->forward_base());
                
                uint_fast8_t expansion_idx = expand_to.m_ndim - 1;
                uint_fast8_t expanded_idx = this->m_ndim - 1;

                expansion.m_shape[expansion_idx] = this->m_shape[expanded_idx];
                expansion.m_strides[expansion_idx] = this->m_strides[expanded_idx];
                expansion.m_dim_sizes[expansion_idx] = this->m_dim_sizes[expanded_idx];
                expansion_idx--;
                expanded_idx--;
                expansion.m_shape[expansion_idx] = this->m_shape[expanded_idx];
                expansion.m_strides[expansion_idx] = this->m_strides[expanded_idx];
                expansion.m_dim_sizes[expansion_idx] = this->m_dim_sizes[expanded_idx];

                for(;expansion_idx--, expanded_idx-- > 0;) {
                    if(expand_to.m_shape[expansion_idx] == this->m_shape[expanded_idx]) {
                        expansion.m_strides[expansion_idx] = this->m_strides[expanded_idx];
                        expansion.m_dim_sizes[expansion_idx] = this->m_dim_sizes[expanded_idx];
                    }
                }

                return expansion;
            }

            const NDArray<T> axes_reorder(const Axes &axes) const noexcept {
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

        public:
            // getters
            inline const T* data() const noexcept {
                return this->m_data;
            }

            inline T* data() noexcept {
                return this->m_data;
            }

            inline const NDArray<T>* base() const noexcept {
                return this->m_base;
            }

            inline T& operator[](uint_fast64_t index) noexcept {
                return this->m_data[index];
            }

            inline const T& operator[](uint_fast64_t index) const noexcept {
                return this->m_data[index];
            }

            // arithmetical functions
            template <typename T2>
            inline NDArray& add_eq(const NDArray<T2> &rhs) noexcept {
                Impl::add_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& add_eq(T value) noexcept {
                Impl::add_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& add(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::add(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& add(TR value, NDArray<TR> &out) const noexcept {
                Impl::add(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> add(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->add(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> add(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->add(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> add(const NDArray<T2> &rhs) const noexcept {
                return this->template add<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& subtract_eq(const NDArray<T2> &rhs) noexcept {
                Impl::subtract_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& subtract_eq(T value) noexcept {
                Impl::subtract_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& subtract(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::subtract(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& subtract(TR value, NDArray<TR> &out) const noexcept {
                Impl::subtract(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> subtract(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->subtract(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> subtract(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->subtract(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> subtract(const NDArray<T2> &rhs) const noexcept {
                return this->template subtract<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& multiply_eq(const NDArray<T2> &rhs) noexcept {
                Impl::multiply_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& multiply_eq(T value) noexcept {
                Impl::multiply_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& multiply(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::multiply(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& multiply(TR value, NDArray<TR> &out) const noexcept {
                Impl::multiply(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> multiply(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->multiply(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> multiply(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->multiply(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> multiply(const NDArray<T2> &rhs) const noexcept {
                return this->template multiply<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& divide_eq(const NDArray<T2> &rhs) noexcept {
                Impl::divide_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& divide_eq(T value) noexcept {
                Impl::divide_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& divide(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::divide(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& divide(TR value, NDArray<TR> &out) const noexcept {
                Impl::divide(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> divide(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->divide(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> divide(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->divide(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> divide(const NDArray<T2> &rhs) const noexcept {
                return this->template divide<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_xor_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_xor_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_xor_eq(T value) noexcept {
                Impl::bit_xor_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_xor(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_xor(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_xor(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_xor(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_xor(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->bit_xor(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_xor(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->bit_xor(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_xor(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_xor<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_and_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_and_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_and_eq(T value) noexcept {
                Impl::bit_and_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_and(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_and(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_and(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_and(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_and(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->bit_and(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_and(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->bit_and(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_and(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_and<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_or_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_or_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_or_eq(T value) noexcept {
                Impl::bit_or_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_or(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_or(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_or(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_or(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_or(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->bit_or(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_or(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->bit_or(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_or(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_or<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& shl_eq(const NDArray<T2> &rhs) noexcept {
                Impl::shl_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& shl_eq(T value) noexcept {
                Impl::shl_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shl(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::shl(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shl(TR value, NDArray<TR> &out) const noexcept {
                Impl::shl(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> shl(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->shl(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shl(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->shl(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> shl(const NDArray<T2> &rhs) const noexcept {
                return this->template shl<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& shr_eq(const NDArray<T2> &rhs) noexcept {
                Impl::shr_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& shr_eq(T value) noexcept {
                Impl::shr_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shr(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::shr(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shr(TR value, NDArray<TR> &out) const noexcept {
                Impl::shr(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> shr(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->shr(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shr(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->shr(value, out);
                return out;
            }

            inline NDArray& bit_not_eq() noexcept {
                Impl::bit_not_eq(this->m_data, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& bit_not(NDArray<TR> &out) const noexcept {
                Impl::bit_not(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> bit_not() const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->bit_not(out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> shr(const NDArray<T2> &rhs) const noexcept {
                return this->template shr<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& remainder_eq(const NDArray<T2> &rhs) noexcept {
                Impl::remainder_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& remainder_eq(T value) noexcept {
                Impl::remainder_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& remainder(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::remainder(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& remainder(TR value, NDArray<TR> &out) const noexcept {
                Impl::remainder(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> remainder(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->remainder(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> remainder(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->remainder(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> remainder(const NDArray<T2> &rhs) const noexcept {
                return this->template remainder<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& power_eq(const NDArray<T2> &rhs) noexcept {
                Impl::power_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& power_eq(T value) noexcept {
                Impl::power_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& power(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::power(this->m_data, this->m_shape == out.m_shape ? *this : this->expansion(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expansion(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& power(TR value, NDArray<TR> &out) const noexcept {
                Impl::power(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expansion(out),
                value, out.m_data, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> power(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->m_shape, rhs.m_shape));
                this->power(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> power(TR value) const noexcept {
                NDArray<TR> out(new TR[this->m_size], *this, nullptr);
                this->power(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> power(const NDArray<T2> &rhs) const noexcept {
                return this->template power<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& matmul(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                using laruen::math::common::min, laruen::math::common::is_pow2;
                using laruen::math::bits::lsb64;

                uint_fast64_t lhs_rows = this->m_shape[this->m_ndim - 2];
                uint_fast64_t rhs_cols = rhs.m_shape.back();
                uint_fast64_t lhs_shared = this->m_shape.back();

                assert(lhs_rows == out.m_shape[out.m_ndim - 2] &&
                rhs_cols == out.m_shape.back() &&
                lhs_shared == rhs.m_shape[rhs.m_ndim - 2]);

                uint_fast8_t depth = min(lsb64(lhs_rows), min(lsb64(rhs_cols), lsb64(lhs_shared)));

                if(is_pow2(lhs_rows) && is_pow2(rhs_cols) && is_pow2(lhs_shared)) {
                    depth--;
                }

                Impl::matmul(this->m_data,
                std::equal(out.m_shape.begin(), out.m_shape.end() - 2, this->m_shape.begin())
                ? *this : this->matmul_expansion(out),
                rhs.m_data,
                std::equal(out.m_shape.begin(), out.m_shape.end() - 2, rhs.m_shape.begin())
                ? rhs : rhs.matmul_expansion(out),
                out.m_data, out, depth);

                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> matmul(const NDArray<T2> &rhs) const {
                NDArray<TR> out(laruen::ndlib::utils::matmul_broadcast(this->m_shape, rhs.m_shape));
                this->matmul(rhs, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> matmul(const NDArray<T2> &rhs) const {
                return this->matmul<types::result_type_t<T, T2>, T2>(rhs);
            }

            // arithmetical operators
            template <typename T2>
            inline NDArray& operator+=(const NDArray<T2> &rhs) {
                return this->add_eq(rhs);
            }

            inline NDArray& operator+=(T value) noexcept {
                return this->add_eq(value);
            }

            template <typename T2>
            inline auto operator+(const NDArray<T2> &rhs) const {
                return this->add(rhs);
            }

            inline NDArray<T> operator+(T value) const noexcept {
                return this->add(value);
            }

            template <typename T2>
            inline NDArray& operator-=(const NDArray<T2> &rhs) {
                return this->subtract_eq(rhs);
            }

            inline NDArray& operator-=(T value) noexcept {
                return this->subtract_eq(value);
            }
            
            template <typename T2>
            inline auto operator-(const NDArray<T2> &rhs) const {
                return this->subtract(rhs);
            }

            inline NDArray<T> operator-(T value) const noexcept {
                return this->subtract(value);
            }

            template <typename T2>
            inline NDArray& operator*=(const NDArray<T2> &rhs) {
                return this->multiply_eq(rhs);
            }

            inline NDArray& operator*=(T value) noexcept {
                return this->multiply_eq(value);
            }

            template <typename T2>
            inline auto operator*(const NDArray<T2> &rhs) const {
                return this->multiply(rhs);
            }

            inline NDArray<T> operator*(T value) const noexcept {
                return this->multiply(value);
            }

            template <typename T2>
            inline NDArray& operator/=(const NDArray<T2> &rhs) {
                return this->divide_eq(rhs);
            }

            inline NDArray& operator/=(T value) noexcept {
                return this->divide_eq(value);
            }

            template <typename T2>
            inline auto operator/(const NDArray<T2> &rhs) const {
                return this->divide(rhs);
            }

            inline NDArray<T> operator/(T value) const noexcept {
                return this->divide(value);
            }
            
            template <typename T2>
            inline NDArray& operator^=(const NDArray<T2> &rhs) {
                return this->bit_xor_eq(rhs);
            }

            inline NDArray& operator^=(T value) noexcept {
                return this->bit_xor_eq(value);
            }

            template <typename T2>
            inline auto operator^(const NDArray<T2> &rhs) const {
                return this->bit_xor(rhs);
            }

            inline NDArray<T> operator^(T value) const noexcept {
                return this->bit_xor(value);
            }
            
            template <typename T2>
            inline NDArray& operator&=(const NDArray<T2> &rhs) {
                return this->bit_and_eq(rhs);
            }

            inline NDArray& operator&=(T value) noexcept {
                return this->bit_and_eq(value);
            }

            template <typename T2>
            inline auto operator&(const NDArray<T2> &rhs) const {
                return this->bit_and(rhs);
            }

            inline NDArray<T> operator&(T value) const noexcept {
                return this->bit_and(value);
            }

            template <typename T2>
            inline NDArray& operator|=(const NDArray<T2> &rhs) {
                return this->bit_or_eq(rhs);
            }
            
            inline NDArray& operator|=(T value) noexcept {
                return this->bit_or_eq(value);
            }

            template <typename T2>
            inline auto operator|(const NDArray<T2> &rhs) const {
                return this->bit_or(rhs);
            }

            inline NDArray<T> operator|(T value) const noexcept {
                return this->bit_or(value);
            }
            
            template <typename T2>
            inline NDArray& operator<<=(const NDArray<T2> &rhs) {
                return this->shl_eq(rhs);
            }

            inline NDArray& operator<<=(T value) noexcept {
                return this->shl_eq(value);
            }
            template <typename T2>

            inline auto operator<<(const NDArray<T2> &rhs) const {
                return this->shl(rhs);
            }

            inline NDArray<T> operator<<(T value) const noexcept {
                return this->shl(value);
            }
            
            template <typename T2>
            inline NDArray& operator>>=(const NDArray<T2> &rhs) {
                return this->shr_eq(rhs);
            }

            inline NDArray& operator>>=(T value) noexcept {
                return this->shr_eq(value);
            }

            template <typename T2>
            inline auto operator>>(const NDArray<T2> &rhs) const {
                return this->shr(rhs);
            }

            inline NDArray<T> operator>>(T value) const noexcept {
                return this->shr(value);
            }
            
            inline NDArray& operator~() noexcept {
                return this->bit_not();
            }

            inline NDArray& operator%=(T value) noexcept {
                return this->remainder_eq(value);
            }

            template <typename T2>
            inline NDArray& operator%=(const NDArray<T2> &rhs) {
                return this->remainder_eq(rhs);
            }

            template <typename T2>
            inline auto operator%(const NDArray<T2> &rhs) const {
                return this->remainder(rhs);
            }

            inline NDArray<T> operator%(T value) const noexcept {
                return this->remainder(value);
            }

            // more utility functions
            inline ArrayBase& arraybase() noexcept {
                return *this;
            }

            inline const ArrayBase& arraybase() const noexcept {
                return *this;
            }

            inline const NDArray<T>* forward_base() const noexcept {
                return this->m_base ? this->m_base : this;
            }
            
            inline NDArray<T> view() noexcept {
                return NDArray<T>(this->m_data, this->m_shape,
                this->m_strides, this->m_size, this->m_ndim, this->forward_base());
            }

            inline const NDArray<T> view() const noexcept {
                return NDArray<T>(this->m_data, this->m_shape,
                this->m_strides, this->m_size, this->m_ndim, this->forward_base());
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const NDArray &ndarray) noexcept {
                return stream << ndarray.str();
            }
    };
};

#endif