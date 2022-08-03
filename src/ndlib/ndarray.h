
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
#include "src/math/constants.h"


namespace laruen::ndlib {

    template <typename T = float64_t>
    class NDArray : public ArrayBase {

        template <typename> friend class NDArray;
        friend struct Impl;

        private:
            T *data_;
            const NDArray<T> *base_ = nullptr;

        public:
            typedef T DType;

            ~NDArray() {
                if(!this->base_) {
                    delete[] this->data_;
                }
            }

            // constructors and assignment operators
            NDArray() noexcept : ArrayBase(), data_(nullptr)
            {}

            NDArray(std::initializer_list<T> init_list) noexcept
            : ArrayBase(Shape{init_list.size()}, Strides{1}, Strides{init_list.size()}, init_list.size(), 1, true),
            data_(new T[init_list.size()])
            {
                NDIter iter(this->data_, *this);

                for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
                    iter.next() = *list_ptr;
                }
            }

            NDArray(std::initializer_list<T> init_list, const Shape &shape) noexcept
            : NDArray(shape)
            {
                assert(init_list.size() == this->size_);

                NDIter iter(this->data_, *this);

                for(const T *list_ptr = init_list.begin();list_ptr != init_list.end();list_ptr++) {
                    iter.next() = *list_ptr;
                }
            }

            NDArray(T *data, const Shape &shape, const Strides &strides, const Strides &dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig, const NDArray *base = nullptr) noexcept
            : ArrayBase(shape, strides, dim_sizes, size, ndim, contig), data_(data), base_(base)
            {}

            NDArray(T *data, Shape &&shape, Strides &&strides, Strides &&dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig, const NDArray *base = nullptr) noexcept
            : ArrayBase(std::move(shape), std::move(strides), std::move(dim_sizes), size, ndim, contig),
            data_(data), base_(base)
            {}
            
            explicit NDArray(const Shape &shape) noexcept
            : ArrayBase(shape), data_(new T[this->size_])
            {}
            
            NDArray(const Shape &shape, T value) noexcept
            : NDArray(shape) {
                this->fill(value);
            }
            
            NDArray(T *data, const ArrayBase &arraybase, const NDArray<T> *base = nullptr) noexcept
            : ArrayBase(arraybase), data_(data), base_(base)
            {}
            
            NDArray(const NDArray &ndarray) noexcept 
            : NDArray(new T[ndarray.size_], ndarray) {
                this->copy_data_from(ndarray);
            }
            
            NDArray(NDArray &&ndarray) noexcept
            : ArrayBase(std::move(ndarray)), data_(ndarray.data_) {
                ndarray.data_ = nullptr;
            }
            
            explicit NDArray(const Range<T> &range) noexcept
            : NDArray(Shape{laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step)})
            {
                T value = range.start;

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    this->data_[i] = value;
                    value += range.step;
                }
            }
            
            NDArray(const Shape &shape, const Range<T> &range)
            : NDArray(shape)
            {
                if(laruen::ndlib::utils::ceil_index((range.end - range.start) / range.step) != this->size_) {
                    throw std::invalid_argument("shape size does not match range");
                }

                T value = range.start;

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    this->data_[i] = value;
                    value += range.step;
                }
            }
            
            NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept
            : ArrayBase(axes.size(), axes.size() > 0)
            {
                uint_fast8_t axis;
                uint_fast64_t stride = 1;

                for(uint_fast8_t i = this->ndim_;i-- > 0;) {
                    axis = axes[i];
                    this->shape_[i] = arraybase.shape_[axis];
                    this->strides_[i] = stride;
                    stride *= this->shape_[i];
                    this->dim_sizes_[i] = stride;
                    this->size_ *= this->shape_[i];
                }

                this->data_ = new T[this->size_];
                this->contig_ = false;
            }
            
            NDArray(NDArray<T> &ndarray, const SliceRanges &ranges) noexcept
            : NDArray(ndarray.data_, ndarray, ndarray.forward_base())
            {
                uint_fast8_t ndim = ranges.size();
                float64_t size_ratio = 1;

                for(uint_fast8_t dim = 0;dim < ndim;dim++) {
                    size_ratio *= this->shape_[dim];
                    this->data_ += ranges[dim].start * this->strides_[dim];
                    this->strides_[dim] *= ranges[dim].step;
                    this->shape_[dim] = laruen::ndlib::utils::ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
                    this->dim_sizes_[dim] = this->shape_[dim] * this->strides_[dim];
                    size_ratio /= this->shape_[dim];
                }

                this->size_ /= size_ratio;
                this->contig_ = false;
            }
            
            template <typename T2>
            NDArray(const NDArray<T2> &ndarray) noexcept
            : NDArray<T>(new T[ndarray.size_], ndarray) {
                this->copy_data_from(ndarray);
            }
            
            template <typename T2>
            NDArray(NDArray<T2> &&ndarray) noexcept
            : ArrayBase(std::move(ndarray)), data_(new T[ndarray.size_]) {
                this->copy_data_from(ndarray);
            }

            NDArray& operator=(const NDArray &ndarray) noexcept {
                if(this == &ndarray) {
                    return *this;
                }

                if(this->size_ != ndarray.size_) {
                    if(!this->base_) {
                        delete[] this->data_;
                    }
                    this->data_ = new T[ndarray.size_];
                    this->contig_ = true;
                    // base update is required - 
                    // creating new data means
                    // 'this' object is the owner
                    this->base_ = nullptr;
                }

                this->shape_ = ndarray.shape_;
                this->strides_ = ndarray.strides_;
                this->dim_sizes_ = ndarray.dim_sizes_;
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;

                this->copy_data_from(ndarray);

                return *this;
            }
            
            NDArray& operator=(NDArray &&ndarray) noexcept {
                if(this == &ndarray) {
                    return *this;
                }

                if(!this->base_) {
                    delete[] this->data_;
                }
                
                this->shape_ = std::move(ndarray.shape_);
                this->strides_ = std::move(ndarray.strides_);
                this->dim_sizes_ = std::move(ndarray.dim_sizes_);
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;
                this->contig_ = ndarray.contig_;
                // base update required since memory space
                // of data is not ours
                this->base_ = ndarray.base_;
                
                this->data_ = ndarray.data_;
                ndarray.data_ = nullptr;

                return *this;
            }
            
            template <typename T2>
            NDArray& operator=(const NDArray<T2> &ndarray) noexcept {
                // needs fix
                if(this->size_ != ndarray.size_) {
                    if(!this->base_) {
                        delete[] this->data_;
                    }
                    this->data_ = new T[ndarray.size_];
                    this->contig_ = true;
                    // base update is required - 
                    // creating new data means
                    // 'this' object is the owner
                    this->base_ = nullptr;
                }

                this->shape_ = ndarray.shape_;
                this->strides_ = ndarray.strides_;
                this->dim_sizes_=  ndarray.dim_sizes_;
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;

                this->copy_data_from(ndarray);

                return *this;
            }
            
            template <typename T2>
            NDArray& operator=(NDArray<T2> &&ndarray) noexcept {
                this->data_ = new T[ndarray.size_];
                this->shape_ = std::move(ndarray.shape_);
                this->strides_ = std::move(ndarray.strides_);
                this->dim_sizes_ = std::move(ndarray.dim_sizes_);
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;
                this->contig_ = true;
                // base change needs to be done -
                // new data means 'this' object
                // is the owner of the data
                this->base_ = nullptr;

                this->copy_data_from(ndarray);

                return *this;
            }

            // utility functions
            template <typename T2>
            void copy_data_from(const NDArray<T2> &ndarray) noexcept {
                NDIter to(this->data_, *this);
                NDIter from(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    to.next() = from.next();
                }
            }
            
            void fill(T value) noexcept {
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    iter.next() = value;
                }
            }

            // computational functions on the array
            template <typename TR>
            NDArray<TR>& sum(const Axes &axes, NDArray<TR> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter this_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T sum;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    sum = 0;

                    for(uint_fast64_t j = 0;j < sample_size;j++) {
                        sum += this_iter.next();
                    }
                    out_iter.next() = sum;
                }

                return out;
            }

            NDArray<T> sum(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->ndim_));
                this->sum(axes, out);
                return out;
            }

            T sum() const noexcept {
                T sum = 0;
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    sum += iter.next();
                }

                return sum;
            }

            template <typename TR>
            NDArray<TR>& max(const Axes &axes, NDArray<TR> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter this_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T max;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    max = this_iter.next();
                    
                    for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                        max = laruen::math::common::max(max, this_iter.next());
                    }
                    out_iter.next() = max;
                }

                return out;
            }

            NDArray<T> max(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->ndim_));
                this->max(axes, out);
                return out;
            }
            
            T max() const noexcept {
                NDIter iter(this->data_, *this);
                T max = iter.next();

                for(uint_fast64_t i = 1;i < this->size_;i++) {
                    max = laruen::math::common::max(max, iter.next());
                }

                return max;
            }
            
            NDArray<uint_fast64_t>& indices_max(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T max;
                T current;
                T *ptr_max;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
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
                    out_iter.next() = ptr_max - this->data_;
                }

                return out;
            }

            NDArray<uint_fast64_t> indices_max(const Axes &axes) const noexcept {
                NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->ndim_));
                this->indices_max(axes, out);
                return out;
            }

            uint_fast64_t index_max() const noexcept {
                NDIter iter(this->data_, *this);
                T value;
                T max = iter.next();
                uint_fast64_t index_max = 0;

                for(uint_fast64_t i = 1;i < this->size_;i++) {
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

                NDIter out_iter(out.data_, out);
                NDIter this_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T min;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    min = this_iter.next();
                    
                    for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                        min = laruen::math::common::min(min, this_iter.next());
                    }
                    out_iter.next() = min;
                }

                return out;
            }
            
            NDArray<T> min(const Axes &axes) const noexcept {
                NDArray<T> out(*this, laruen::ndlib::utils::compress_axes(axes, this->ndim_));
                this->min(axes, out);
                return out;
            }

            T min() const noexcept {
                NDIter iter(this->data_, *this);
                T min = iter.next();

                for(uint_fast64_t i = 1;i < this->size_;i++) {
                    min = laruen::math::common::min(min, iter.next());
                }

                return min;
            }
            
            NDArray<uint_fast64_t>& indices_min(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T min;
                T current;
                T *ptr_min;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
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
                    out_iter.next() = ptr_min - this->data_;
                }

                return out;
            }

            NDArray<uint_fast64_t> indices_min(const Axes &axes) const noexcept {
                NDArray<uint_fast64_t> out(*this, laruen::ndlib::utils::compress_axes(axes, this->ndim_));
                this->indices_min(axes, out);
                return out;
            }

            uint_fast64_t index_min() const noexcept {
                NDIter iter(this->data_, *this);
                T value;
                T min = iter.next();
                uint_fast64_t index_min = 0;

                for(uint_fast64_t i = 1;i < this->size_;i++) {
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
                return this->data_[this->ravel_ndindex(ndindex)];
            }

            const T& operator[](const NDIndex &ndindex) const noexcept {
                return this->data_[this->ravel_ndindex(ndindex)];
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
                bool eq = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && eq;i++) {
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
                bool ge = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && ge;i++) {
                    ge = (lhs_iter.next() >= rhs_iter.next());
                }

                return ge;
            }
            
            template <typename T2>
            bool operator<=(const NDArray<T2> &ndarray) const noexcept {
                bool le = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && le;i++) {
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
                Shape t_shape(this->ndim_);
                Strides t_strides(this->ndim_);
                Strides t_dim_sizes(this->ndim_);

                uint_fast8_t f = 0;
                uint_fast8_t b = this->ndim_ - 1;
                uint_fast8_t mid = this->ndim_ >> 1;

                t_shape[mid] = this->shape_[mid];
                t_strides[mid] = this->strides_[mid];
                t_dim_sizes[mid] = this->dim_sizes_[mid];


                for(;f < mid;f++, b--) {
                    t_shape[f] = this->shape_[b];
                    t_shape[b] = this->shape_[f];
                    t_strides[f] = this->strides_[b];
                    t_strides[b] = this->strides_[f];
                    t_dim_sizes[f] = this->dim_sizes_[b];
                    t_dim_sizes[b] = this->dim_sizes_[f];
                }

                return NDArray<T>(this->data_, std::move(t_shape),
                std::move(t_strides), std::move(t_dim_sizes), this->size_, this->ndim_, this->forward_base());
            }

            const NDArray<T> transpose() const noexcept {
                Shape t_shape(this->ndim_);
                Strides t_strides(this->ndim_);
                Strides t_dim_sizes(this->ndim_);

                uint_fast8_t f = 0;
                uint_fast8_t b = this->ndim_ - 1;
                uint_fast8_t mid = this->ndim_ >> 1;

                t_shape[mid] = this->shape_[mid];
                t_strides[mid] = this->strides_[mid];
                t_dim_sizes[mid] = this->dim_sizes_[mid];


                for(;f < mid;f++, b--) {
                    t_shape[f] = this->shape_[b];
                    t_shape[b] = this->shape_[f];
                    t_strides[f] = this->strides_[b];
                    t_strides[b] = this->strides_[f];
                    t_dim_sizes[f] = this->dim_sizes_[b];
                    t_dim_sizes[b] = this->dim_sizes_[f];
                }

                return NDArray<T>(this->data_, std::move(t_shape),
                std::move(t_strides), std::move(t_dim_sizes), this->size_, this->ndim_, this->forward_base());
            }

            NDArray<T> view_reshape(const Shape &shape) noexcept {
                NDArray<T> view = this->view();
                view.reshape(shape);
                return view;
            }

            const NDArray<T> view_reshape(const Shape &shape) const noexcept {
                NDArray<T> view = this->view();
                view.reshape(shape);
                return view;
            }

            NDArray<T> copy_reshape(const Shape &shape) noexcept {
                NDArray<T> copy(shape);
                assert(copy.size_ == this->size_);
                copy.copy_data_from(*this);
                return copy;
            }
            
            const NDArray<T> copy_reshape(const Shape &shape) const noexcept {
                NDArray<T> copy(shape);
                assert(copy.size_ == this->size_);
                copy.copy_data_from(*this);
                return copy;
            }

            NDArray<T> new_reshape(const Shape &shape) noexcept {
                return this->contig_ ? this->view_reshape(shape) : this->copy_reshape(shape);
            }

            const NDArray<T> new_reshape(const Shape &shape) const noexcept {
                return this->contig_ ? this->view_reshape(shape) : this->copy_reshape(shape);
            }

            // string function
            std::string str() const noexcept {
                NDIndex ndindex(this->ndim_, 0);
                uint_fast64_t index = 0;
                uint_fast8_t dim = 0;
                std::string str;

                if(!this->size_) {
                    str.push_back('[');
                    str.push_back(']');
                    return str;
                }

                str.reserve(this->size_ * (this->ndim_ / 2) * 19);

                for(uint_fast64_t i = 0;i < this->size_ - 1;i++) {
                    if(!ndindex[this->ndim_ - 1]) {
                        str += std::string(dim, ' ') + std::string(this->ndim_ - dim, '[');
                    }

                    str += std::to_string(this->data_[index]);
                    ndindex[this->ndim_ - 1]++;
                    index += this->strides_[this->ndim_ - 1];

                    for(dim = this->ndim_;dim-- > 1 && ndindex[dim] >= this->shape_[dim];) {
                        ndindex[dim] = 0;
                        ndindex[dim - 1]++;
                        index += this->strides_[dim - 1] - shape_[dim] * this->strides_[dim];
                        str.push_back(']');
                    }
                    dim++;

                    if(dim == this->ndim_) {
                        str.push_back(',');
                        str.push_back(' ');
                    }

                    str += std::string(this->ndim_ - dim, '\n');
                }

                if(!ndindex[this->ndim_ - 1]) {
                        str += std::string(dim, ' ') + std::string(this->ndim_ - dim, '[');
                }

                str += std::to_string(this->data_[index]);
                str += std::string(this->ndim_, ']');

                return str;
            }

        private:
            // private utility functions
            template <typename T2>
            const NDArray<T> expansion(const NDArray<T2> &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expansion */
                
                NDArray<T> expansion(this->data_, Shape(expand_to.shape_), Strides(expand_to.ndim_, 0),
                Strides(expand_to.ndim_, 0), expand_to.size_, expand_to.ndim_, false, this->forward_base());
                
                uint_fast8_t expansion_idx = expansion.ndim_ - this->ndim_;

                for(uint_fast8_t expanded_idx = 0;expanded_idx < this->ndim_;expanded_idx++, expansion_idx++) {
                    if(expansion.shape_[expansion_idx] == this->shape_[expanded_idx]) {
                        expansion.strides_[expansion_idx] = this->strides_[expanded_idx];
                        expansion.dim_sizes_[expansion_idx] = this->dim_sizes_[expanded_idx];
                    }
                }

                return expansion;
            }

            template <typename T2>
            const NDArray<T> matmul_expansion(const NDArray<T2> &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expand_to */
                
                NDArray<T> expansion(this->data_, Shape(expand_to.shape_), Strides(expand_to.ndim_, 0),
                Strides(expand_to.ndim_, 0), expand_to.size_, expand_to.ndim_, false, this->forward_base());
                
                uint_fast8_t expansion_idx = expand_to.ndim_ - 1;
                uint_fast8_t expanded_idx = this->ndim_ - 1;

                expansion.shape_[expansion_idx] = this->shape_[expanded_idx];
                expansion.strides_[expansion_idx] = this->strides_[expanded_idx];
                expansion.dim_sizes_[expansion_idx] = this->dim_sizes_[expanded_idx];
                expansion_idx--;
                expanded_idx--;
                expansion.shape_[expansion_idx] = this->shape_[expanded_idx];
                expansion.strides_[expansion_idx] = this->strides_[expanded_idx];
                expansion.dim_sizes_[expansion_idx] = this->dim_sizes_[expanded_idx];

                for(;expansion_idx--, expanded_idx-- > 0;) {
                    if(expand_to.shape_[expansion_idx] == this->shape_[expanded_idx]) {
                        expansion.strides_[expansion_idx] = this->strides_[expanded_idx];
                        expansion.dim_sizes_[expansion_idx] = this->dim_sizes_[expanded_idx];
                    }
                }

                return expansion;
            }

            const NDArray<T> axes_reorder(const Axes &axes) const noexcept {
                NDArray<T> reorder(this->data_, Shape(this->ndim_), Strides(this->ndim_),
                Strides(this->ndim_), this->size_, this->ndim_, false, this->forward_base());

                uint_fast8_t axes_added = 0;
                uint_fast8_t dest_idx = reorder.ndim_ - 1;
                
                for(uint_fast8_t i = axes.size();i-- > 0;) {
                    reorder.shape_[dest_idx] = this->shape_[axes[i]];
                    reorder.strides_[dest_idx] = this->strides_[axes[i]];
                    reorder.dim_sizes_[dest_idx] = this->dim_sizes_[axes[i]];
                    axes_added |= 1 << axes[i];
                    dest_idx--;
                }

                for(uint_fast8_t i = reorder.ndim_;i-- > 0;) {
                    if(!(axes_added & (1 << i))) {
                        reorder.shape_[dest_idx] = this->shape_[i];
                        reorder.strides_[dest_idx] = this->strides_[i];
                        reorder.dim_sizes_[dest_idx] = this->dim_sizes_[i];
                        dest_idx--; 
                    }
                }

                return reorder;
            }

        public:
            // getters
            inline const T* data() const noexcept {
                return this->data_;
            }

            inline T* data() noexcept {
                return this->data_;
            }

            inline const NDArray<T>* base() const noexcept {
                return this->base_;
            }

            inline T& operator[](uint_fast64_t index) noexcept {
                return this->data_[index];
            }

            inline const T& operator[](uint_fast64_t index) const noexcept {
                return this->data_[index];
            }

            // arithmetical functions
            template <typename T2>
            inline NDArray& add_eq(const NDArray<T2> &rhs) noexcept {
                Impl::add_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& add_eq(T value) noexcept {
                Impl::add_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& add(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::add(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& add(TR value, NDArray<TR> &out) const noexcept {
                Impl::add(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> add(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->add(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> add(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->add(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> add(const NDArray<T2> &rhs) const noexcept {
                return this->template add<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& subtract_eq(const NDArray<T2> &rhs) noexcept {
                Impl::subtract_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& subtract_eq(T value) noexcept {
                Impl::subtract_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& subtract(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::subtract(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& subtract(TR value, NDArray<TR> &out) const noexcept {
                Impl::subtract(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> subtract(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->subtract(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> subtract(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->subtract(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> subtract(const NDArray<T2> &rhs) const noexcept {
                return this->template subtract<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& multiply_eq(const NDArray<T2> &rhs) noexcept {
                Impl::multiply_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& multiply_eq(T value) noexcept {
                Impl::multiply_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& multiply(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::multiply(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& multiply(TR value, NDArray<TR> &out) const noexcept {
                Impl::multiply(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> multiply(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->multiply(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> multiply(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->multiply(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> multiply(const NDArray<T2> &rhs) const noexcept {
                return this->template multiply<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& divide_eq(const NDArray<T2> &rhs) noexcept {
                Impl::divide_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& divide_eq(T value) noexcept {
                Impl::divide_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& divide(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::divide(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& divide(TR value, NDArray<TR> &out) const noexcept {
                Impl::divide(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> divide(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->divide(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> divide(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->divide(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> divide(const NDArray<T2> &rhs) const noexcept {
                return this->template divide<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_xor_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_xor_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_xor_eq(T value) noexcept {
                Impl::bit_xor_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_xor(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_xor(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_xor(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_xor(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_xor(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_xor(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_xor(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->bit_xor(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_xor(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_xor<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_and_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_and_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_and_eq(T value) noexcept {
                Impl::bit_and_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_and(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_and(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_and(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_and(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_and(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_and(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_and(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->bit_and(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_and(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_and<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& bit_or_eq(const NDArray<T2> &rhs) noexcept {
                Impl::bit_or_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& bit_or_eq(T value) noexcept {
                Impl::bit_or_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_or(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::bit_or(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_or(TR value, NDArray<TR> &out) const noexcept {
                Impl::bit_or(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> bit_or(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_or(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_or(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->bit_or(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> bit_or(const NDArray<T2> &rhs) const noexcept {
                return this->template bit_or<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& shl_eq(const NDArray<T2> &rhs) noexcept {
                Impl::shl_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& shl_eq(T value) noexcept {
                Impl::shl_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shl(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::shl(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shl(TR value, NDArray<TR> &out) const noexcept {
                Impl::shl(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> shl(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->shl(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shl(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->shl(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> shl(const NDArray<T2> &rhs) const noexcept {
                return this->template shl<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& shr_eq(const NDArray<T2> &rhs) noexcept {
                Impl::shr_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& shr_eq(T value) noexcept {
                Impl::shr_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shr(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::shr(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shr(TR value, NDArray<TR> &out) const noexcept {
                Impl::shr(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> shr(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->shr(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shr(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->shr(value, out);
                return out;
            }

            inline NDArray& bit_not_eq() noexcept {
                Impl::bit_not_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& bit_not(NDArray<TR> &out) const noexcept {
                Impl::bit_not(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> bit_not() const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->bit_not(out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> shr(const NDArray<T2> &rhs) const noexcept {
                return this->template shr<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& remainder_eq(const NDArray<T2> &rhs) noexcept {
                Impl::remainder_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            inline NDArray& remainder_eq(T value) noexcept {
                Impl::remainder_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& remainder(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::remainder(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& remainder(TR value, NDArray<TR> &out) const noexcept {
                Impl::remainder(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> remainder(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->remainder(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> remainder(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->remainder(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> remainder(const NDArray<T2> &rhs) const noexcept {
                return this->template remainder<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& power_eq(const NDArray<T2> &rhs) noexcept {
                Impl::power_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            template <typename T2>
            inline NDArray& power_eq(T2 value) noexcept {
                Impl::power_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& power(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::power(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& power(T2 value, NDArray<TR> &out) const noexcept {
                Impl::power(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> power(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->power(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> power(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->power(value, out);
                return out;
            }

            template <typename TR, typename T2, typename = std::enable_if_t<!std::is_same_v<TR, T2>>>
            inline NDArray<TR> power(T2 value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->power(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> power(const NDArray<T2> &rhs) const noexcept {
                return this->template power<types::result_type_t<T, T2>, T2>(rhs);
            }

            template <typename T2>
            inline NDArray& inverse_power_eq(const NDArray<T2> &rhs) noexcept {
                Impl::inverse_power_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : rhs.expansion(*this));
                return *this;
            }

            template <typename T2>
            inline NDArray& inverse_power_eq(T2 value) noexcept {
                Impl::inverse_power_eq(this->data_, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& inverse_power(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                Impl::inverse_power(this->data_, this->shape_ == out.shape_ ? *this : this->expansion(out),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : rhs.expansion(out),
                out.data_, out);
                return out;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& inverse_power(T2 value, NDArray<TR> &out) const noexcept {
                Impl::inverse_power(this->data_,
                this->shape_ == out.shape_ ? *this : this->expansion(out),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> inverse_power(const NDArray<T2> &rhs) const noexcept {
                NDArray<TR> out(laruen::ndlib::utils::broadcast(this->shape_, rhs.shape_));
                this->inverse_power(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_power(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->inverse_power(value, out);
                return out;
            }

            template <typename TR, typename T2, typename = std::enable_if_t<!std::is_same_v<TR, T2>>>
            inline NDArray<TR> inverse_power(T2 value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, nullptr);
                this->inverse_power(value, out);
                return out;
            }

            template <typename T2>
            inline NDArray<types::result_type_t<T, T2>> inverse_power(const NDArray<T2> &rhs) const noexcept {
                return this->template inverse_power<types::result_type_t<T, T2>, T2>(rhs);
            }

            inline NDArray& exp_eq() noexcept {
                return this->inverse_power_eq(laruen::math::constants::EULERS_NUMBER);
            }

            template <typename TR>
            inline NDArray<TR>& exp(NDArray<TR> &out) const {
                return this->inverse_power(laruen::math::constants::EULERS_NUMBER, out);
            }

            template <typename TR = types::result_type_t<T, std::remove_const_t<decltype(laruen::math::constants::EULERS_NUMBER)>>>
            inline NDArray<TR> exp() const noexcept {
                return this->template inverse_power<TR>(laruen::math::constants::EULERS_NUMBER);
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& matmul(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                using laruen::math::common::min, laruen::math::common::is_pow2;
                using laruen::math::bits::lsb64;

                uint_fast64_t lhs_rows = this->shape_[this->ndim_ - 2];
                uint_fast64_t rhs_cols = rhs.shape_.back();
                uint_fast64_t lhs_shared = this->shape_.back();

                assert(lhs_rows == out.shape_[out.ndim_ - 2] &&
                rhs_cols == out.shape_.back() &&
                lhs_shared == rhs.shape_[rhs.ndim_ - 2]);

                uint_fast8_t depth = min(lsb64(lhs_rows), min(lsb64(rhs_cols), lsb64(lhs_shared)));

                if(is_pow2(lhs_rows) && is_pow2(rhs_cols) && is_pow2(lhs_shared)) {
                    depth--;
                }

                Impl::matmul(this->data_,
                std::equal(out.shape_.begin(), out.shape_.end() - 2, this->shape_.begin())
                ? *this : this->matmul_expansion(out),
                rhs.data_,
                std::equal(out.shape_.begin(), out.shape_.end() - 2, rhs.shape_.begin())
                ? rhs : rhs.matmul_expansion(out),
                out.data_, out, depth);

                return out;
            }

            template <typename TR, typename T2>
            inline NDArray<TR> matmul(const NDArray<T2> &rhs) const {
                NDArray<TR> out(laruen::ndlib::utils::matmul_broadcast(this->shape_, rhs.shape_));
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
                return this->base_ ? this->base_ : this;
            }
            
            inline NDArray<T> view() noexcept {
                return NDArray<T>(this->data_, this->shape_, this->strides_,
                this->dim_sizes_, this->size_, this->ndim_, this->contig_, this->forward_base());
            }

            inline const NDArray<T> view() const noexcept {
                return NDArray<T>(this->data_, this->shape_, this->strides,
                this->dim_sizes_, this->size_, this->ndim_, this->contig_, this->forward_base());
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const NDArray &ndarray) noexcept {
                return stream << ndarray.str();
            }
    };
};

#endif