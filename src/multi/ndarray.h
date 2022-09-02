
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
#include <random>
#include "src/multi/utils.h"
#include "src/multi/types.h"
#include "src/multi/nditer.h"
#include "src/multi/array_base.h"
#include "src/multi/range.h"
#include "src/multi/type_selection.h"
#include "src/multi/impl.h"
#include "src/multi/rng.h"
#include "src/math/common.h"
#include "src/math/bits.h"
#include "src/math/constants.h"


namespace laruen::multi {

    template <typename T = float32_t>
    class NDArray : public ArrayBase {

        template <typename> friend class NDArray;
        friend struct Impl;

        private:
            // *** member variables are mutable to allow editing of "view"s ***
            mutable T *data_;
            mutable bool data_owner_;

        public:
            typedef T DType;

            ~NDArray() {
                if(this->data_owner_) {
                    delete[] this->data_;
                }
            }

            // constructors and assignment operators
            NDArray() noexcept : ArrayBase(), data_(nullptr), data_owner_(true)
            {}

            NDArray(std::initializer_list<T> init_list) noexcept
            : ArrayBase(Shape{init_list.size()}, Strides{1}, Strides{init_list.size()}, init_list.size(), 1, true),
            data_(new T[init_list.size()]), data_owner_(true)
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

            NDArray(const Shape &shape, T min, T max) noexcept : NDArray<T>(shape) {
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                "Type not supported in random constructor");
                
                if constexpr(std::is_integral_v<T>) {
                    this->randint(min, max);
                }
                else if constexpr(std::is_floating_point_v<T>) {
                    this->rand(min, max);
                }
            }

            NDArray(T *data, const Shape &shape, const Strides &strides, const Strides &dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig, bool data_owner) noexcept
            : ArrayBase(shape, strides, dim_sizes, size, ndim, contig), data_(data), data_owner_(data_owner)
            {}

            NDArray(T *data, Shape &&shape, Strides &&strides, Strides &&dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, bool contig, bool data_owner) noexcept
            : ArrayBase(std::move(shape), std::move(strides), std::move(dim_sizes), size, ndim, contig),
            data_(data), data_owner_(data_owner)
            {}

            NDArray(Shape::const_iterator begin, Shape::const_iterator end) noexcept
            : ArrayBase(begin, end), data_(new T[this->size_]), data_owner_(true)
            {}
            
            explicit NDArray(const Shape &shape) noexcept
            : ArrayBase(shape), data_(new T[this->size_]), data_owner_(true)
            {}

            explicit NDArray(Shape &&shape) noexcept
            : ArrayBase(std::move(shape)), data_(new T[this->size_]), data_owner_(true)
            {}
            
            NDArray(const Shape &shape, T value) noexcept
            : NDArray(shape) {
                this->fill(value);
            }
            
            NDArray(T *data, const ArrayBase &arraybase, bool data_owner) noexcept
            : ArrayBase(arraybase), data_(data), data_owner_(data_owner)
            {}
            
            explicit NDArray(const NDArray &ndarray) noexcept 
            : NDArray(new T[ndarray.size_], ndarray, true) {
                this->copy_data_from(ndarray);
            }
            
            NDArray(NDArray &&ndarray) noexcept
            : ArrayBase(std::move(ndarray)), data_(ndarray.data_), data_owner_(ndarray.data_owner_)
            {
                ndarray.data_ = nullptr;
            }

            NDArray(const NDArray &&ndarray) noexcept
            : ArrayBase(std::move(ndarray.shape_), std::move(ndarray.strides_), std::move(ndarray.dim_sizes_),
            ndarray.size_, ndarray.ndim_, ndarray.contig_),
            data_(ndarray.data_), data_owner_(ndarray.data_owner_)
            {
                ndarray.data_ = nullptr;
                assert(!this->data_owner_);
            }
            
            explicit NDArray(const Range<T> &range) noexcept
            : NDArray(Shape{laruen::multi::utils::ceil_index((range.end - range.start) / range.step)})
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
                if(laruen::multi::utils::ceil_index((range.end - range.start) / range.step) != this->size_) {
                    throw std::invalid_argument("shape size does not match range");
                }

                T value = range.start;

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    this->data_[i] = value;
                    value += range.step;
                }
            }
            
            NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept
            : ArrayBase(axes.size(), axes.size() > 0), data_owner_(true)
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
            : NDArray(ndarray.data_, ndarray, false)
            {
                uint_fast8_t ndim = ranges.size();
                float64_t size_ratio = 1;

                for(uint_fast8_t dim = 0;dim < ndim;dim++) {
                    size_ratio *= this->shape_[dim];
                    this->data_ += ranges[dim].start * this->strides_[dim];
                    this->strides_[dim] *= ranges[dim].step;
                    this->shape_[dim] = laruen::multi::utils::ceil_index((float64_t)(ranges[dim].end - ranges[dim].start) / (float64_t)ranges[dim].step);
                    this->dim_sizes_[dim] = this->shape_[dim] * this->strides_[dim];
                    size_ratio /= this->shape_[dim];
                }

                this->size_ /= size_ratio;
                this->contig_ = false;
            }
            
            template <typename TT>
            explicit NDArray(const NDArray<TT> &ndarray) noexcept
            : NDArray<T>(new T[ndarray.size_], ndarray, true) {
                this->copy_data_from(ndarray);
            }
            
            template <typename TT>
            NDArray(NDArray<TT> &&ndarray) noexcept
            : NDArray<T>(new T[ndarray.size_], ndarray, true) {
                this->copy_data_from(ndarray);
            }

            template <typename TT>
            NDArray(const NDArray<TT> &&ndarray) noexcept
            : NDArray<T>(new T[ndarray.size_], ndarray, true) {
                this->copy_data_from(ndarray);
            }

            NDArray& operator=(const NDArray &ndarray) noexcept {
                if(this == &ndarray) {
                    return *this;
                }

                if(this->size_ != ndarray.size_) {
                    if(this->data_owner_) {
                        delete[] this->data_;
                    }
                    this->data_ = new T[ndarray.size_];
                    this->contig_ = true;
                    this->data_owner_ = true;
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

                if(this->data_owner_) {
                    delete[] this->data_;
                }
                
                this->shape_ = std::move(ndarray.shape_);
                this->strides_ = std::move(ndarray.strides_);
                this->dim_sizes_ = std::move(ndarray.dim_sizes_);
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;
                this->contig_ = ndarray.contig_;
                this->data_owner_ = ndarray.data_owner_;
                
                this->data_ = ndarray.data_;
                ndarray.data_ = nullptr;

                return *this;
            }
            
            const NDArray& operator=(const NDArray &&ndarray) const noexcept {
                if(this == &ndarray) {
                    return *this;
                }

                assert(!ndarray.data_owner_);

                if(this->data_owner_) {
                    delete[] this->data_;
                }
                
                this->shape_ = std::move(ndarray.shape_);
                this->strides_ = std::move(ndarray.strides_);
                this->dim_sizes_ = std::move(ndarray.dim_sizes_);
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;
                this->contig_ = ndarray.contig_;
                this->data_owner_ = ndarray.data_owner_;
                
                this->data_ = ndarray.data_;
                ndarray.data_ = nullptr;

                return *this;
            }
            
            template <typename TT>
            NDArray& operator=(const NDArray<TT> &ndarray) noexcept {
                if(this->size_ != ndarray.size_) {
                    if(this->data_owner_) {
                        delete[] this->data_;
                    }
                    this->data_ = new T[ndarray.size_];
                    this->contig_ = true;
                    this->data_owner_ = true;
                }

                this->shape_ = ndarray.shape_;
                this->strides_ = ndarray.strides_;
                this->dim_sizes_=  ndarray.dim_sizes_;
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;

                this->copy_data_from(ndarray);

                return *this;
            }
            
            template <typename TT>
            NDArray& operator=(const NDArray<TT> &&ndarray) noexcept {
                this->data_ = new T[ndarray.size_];
                this->shape_ = std::move(ndarray.shape_);
                this->strides_ = std::move(ndarray.strides_);
                this->dim_sizes_ = std::move(ndarray.dim_sizes_);
                this->size_ = ndarray.size_;
                this->ndim_ = ndarray.ndim_;
                this->contig_ = true;
                this->data_owner_ = true;

                this->copy_data_from(ndarray);

                return *this;
            }

            // utility functions
            inline void free() noexcept {
                if(this->data_owner_) {
                    delete[] this->data_;
                }

                this->data_ = nullptr;
                this->shape_.clear();
                this->strides_.clear();
                this->dim_sizes_.clear();
                this->size_ = 0;
                this->ndim_ = 0;
            }

            template <typename TT>
            void copy_data_from(const NDArray<TT> &ndarray) noexcept {
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

            template <typename TT>
            inline void swap(NDArray<TT> &rhs) noexcept {
                assert(this->size_ == rhs.size_);
                Impl::swap(this->data_, *this, rhs.data_, rhs);
            }

            // searching algorithms
            int64_t binary_search(T value) const noexcept {
                uint_fast64_t log_index = this->size_ >> 1;
                uint_fast64_t phy_index = this->physical_index(log_index);
                uint_fast8_t max_depth = laruen::math::bits::msb64(this->size_);
                // uint_fast64_t start = 0;
                // uint_fast64_t end = this->physical_index(this->size_);
                
                for(uint_fast8_t depth = 1;depth <= max_depth;) {
                    depth++;
                    
                    if(value == this->data_[phy_index]) {
                        return phy_index;
                    }

                    else if(value < this->data_[phy_index]) {
                        // end = log_index;
                        log_index -= this->size_ >> depth;
                    }
                    
                    else {
                        // start = log_index + 1;
                        log_index += this->size_ >> depth;
                    }

                    phy_index = this->physical_index(log_index);
                }

                phy_index = this->physical_index(log_index - 1);

                return this->data_[phy_index] == value ? phy_index : -1;
            }

            void rand(T min, T max) noexcept {
                // [min, max)
                // - min included, max excluded
                std::uniform_real_distribution<T> dist(min, max);
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    iter.next() = dist(laruen::multi::RNG);
                }
            }

            inline void rand(T max) noexcept {
                this->rand(0, max);
            }

            inline void randint(T min, T max) noexcept {
                // [min, max]
                // - min included, max included
                std::uniform_int_distribution<T> dist(min, max);
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    iter.next() = dist(laruen::multi::RNG);
                }
            }

            inline void randint(T max) noexcept {
                this->randint(0, max);
            }

            void shuffle() noexcept {
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    std::uniform_int_distribution<uint_fast64_t> dist(i, this->size_ - 1);
                    T tmp = iter.current();
                    T *swap_ptr = this->data_ + this->physical_index(dist(laruen::multi::RNG));

                    iter.next() = *swap_ptr;
                    *swap_ptr = tmp;
                }
            }

            void shuffle(uint_fast8_t axis) noexcept {
                ArrayBase view(Shape(this->shape_.begin() + 1, this->shape_.end()),
                Strides(this->strides_.begin() + 1, this->strides_.end()),
                Strides(this->dim_sizes_.begin() + 1, this->dim_sizes_.end()),
                this->size_ / this->shape_[axis], this->ndim_ - 1, !axis && this->contig_);

                T *ptr = this->data_;

                for(uint_fast64_t i = 0;i < this->shape_[axis];i++) {
                    std::uniform_int_distribution<uint_fast64_t> dist(i, this->shape_[axis] - 1);
                    T *swap_ptr = this->data_ + dist(laruen::multi::RNG) * this->strides_[axis];
                    Impl::swap(ptr, view, swap_ptr, view);
                }
            }

            void shuffle(const Axes &axes) noexcept {
                ArrayBase swap_base(*this, utils::compress_axes(axes, this->ndim_));
                ArrayBase iteration_base(*this, axes);

                NDIter iter(this->data_, iteration_base);

                for(uint_fast64_t i = 0;i < iteration_base.size_;i++) {
                    std::uniform_int_distribution<uint_fast64_t> dist(i, iteration_base.size_ - 1);
                    T *swap_ptr = this->data_ + iteration_base.physical_index(dist(laruen::multi::RNG));
                    Impl::swap(iter.ptr, swap_base, swap_ptr, swap_base);
                }
            }

            T random_choice() const noexcept {
                std::uniform_int_distribution<uint_fast64_t> dist(0, this->size_ - 1);
                return this->data_[this->physical_index(dist(laruen::multi::RNG))];
            }

            T random_choice(const NDArray<float64_t> &weights) const noexcept {
                assert(this->size_ == weights.size_);
                
                NDIter iter(this->data_, *this);
                NDIter weights_iter(weights.data_, weights);

                float64_t rand = std::uniform_real_distribution<float64_t>(0, 1)(laruen::multi::RNG);
                float64_t weight_sum = weights_iter.next();

                for(uint_fast64_t i = 0;i < this->size_ && weight_sum < rand;i++) {
                    weight_sum += weights_iter.next();
                    iter.next();
                }

                return iter.current();
            }

            template <typename TR>
            void no_repeat_random_choice(NDArray<TR> &out) const noexcept {
                NDArray<uint_fast64_t> rand_indices(Range(this->size_));
                rand_indices.shuffle();

                NDIter out_iter(out.data_, out);

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    out_iter.next() = this->data_[rand_indices[i]];
                }
            }

            // computational functions on the array
            void abs() noexcept {
                NDIter iter(this->data_, *this);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    iter.next() = laruen::math::common::abs(iter.current());
                }
            }

            NDArray<T> new_abs() noexcept {
                NDArray<T> out(this->shape_);
                NDIter iter(this->data_, *this);
                NDIter out_iter(out.data_, out);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    out_iter.next() = laruen::math::common::abs(iter.next());
                }
                
                return out;
            }

            template <typename TR>
            NDArray<TR>& sum(const Axes &axes, NDArray<TR> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T sum;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    sum = 0;

                    for(uint_fast64_t j = 0;j < sample_size;j++) {
                        sum += src_iter.next();
                    }
                    out_iter.next() = sum;
                }

                return out;
            }

            template <typename TR = T>
            NDArray<TR> sum(const Axes &axes) const noexcept {
                NDArray<TR> out(*this, laruen::multi::utils::compress_axes(axes, this->ndim_));
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
            NDArray<TR>& cumsum(const Axes &axes, NDArray<TR> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = this->axes_size(axes);
                uint_fast64_t iterations = this->size_ / sample_size;
                T sum;

                for(uint_fast64_t i = 0;i < iterations;i++) {
                    sum = 0;

                    for(uint_fast64_t j = 0;j < sample_size;j++) {
                        sum += src_iter.next();
                        out_iter.next() = sum;
                    }
                }

                return out;                
            }
            
            template <typename TR = T>
            NDArray<TR> cumsum(const Axes &axes) const noexcept {
                NDArray<TR> out(this->shape_);
                this->cumsum(axes, out);
                return out;
            }

            template <typename TR>
            NDArray<TR>& cumsum(NDArray<TR> &out) const noexcept {
                NDIter out_iter(out.data_, out);
                NDIter src_iter(this->data_, *this);

                T sum = 0;

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    sum += src_iter.next();
                    out_iter.next() = sum;
                }

                return out;                
            }

            template <typename TR = T>
            NDArray<TR> cumsum() const noexcept {
                NDArray<TR> out(this->shape_);
                this->cumsum(out);
                return out;
            }

            template <typename TT, typename TR>
            NDArray<TR>& maximum(const NDArray<TT> &rhs, NDArray<TR> &out) const {
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(rhs.data_, rhs);
                NDIter out_iter(out.data_, out);
                
                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    out_iter.next() = laruen::math::common::max(lhs_iter.next(), rhs_iter.next());
                }

                return out;
            }

            template <typename TR, typename TT>
            NDArray<TR> maximum(const NDArray<TT> &rhs) const {
                NDArray<TR> out(this->shape_);
                this->maximum(rhs, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> maximum(const NDArray<TT> &rhs) const {
                return this->maximum<types::result_type_t<T, TT>, TT>(rhs);
            }
            
            template <typename TR>
            NDArray<TR>& maximum(T value, NDArray<TR> &out) const {
                NDIter lhs_iter(this->data_, *this);
                NDIter out_iter(out.data_, out);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    out_iter.next() = laruen::math::common::max(lhs_iter.next(), value);
                }

                return out;
            } 

            template <typename TR = T>
            NDArray<TR> maximum(T value) const noexcept {
                NDArray<TR> out(this->shape_);
                this->maximum(value, out);
                return out;
            }

            template <typename TT, typename TR>
            NDArray<TR>& minimum(const NDArray<TT> &rhs, NDArray<TR> &out) const {
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(rhs.data_, rhs);
                NDIter out_iter(out.data_, out);
                
                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    out_iter.next() = laruen::math::common::min(lhs_iter.next(), rhs_iter.next());
                }

                return out;
            }

            template <typename TR, typename TT>
            NDArray<TR> minimum(const NDArray<TT> &rhs) const {
                NDArray<TR> out(this->shape_);
                this->minimum(rhs, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> minimum(const NDArray<TT> &rhs) const {
                return this->minimum<types::result_type_t<T, TT>, TT>(rhs);
            }
            
            template <typename TR>
            NDArray<TR>& minimum(T value, NDArray<TR> &out) const {
                NDIter lhs_iter(this->data_, *this);
                NDIter out_iter(out.data_, out);

                for(uint_fast64_t i = 0;i < this->size_;i++) {
                    out_iter.next() = laruen::math::common::min(lhs_iter.next(), value);
                }

                return out;
            } 

            template <typename TR = T>
            NDArray<TR> minimum(T value) const noexcept {
                NDArray<TR> out(this->shape_);
                this->minimum(value, out);
                return out;
            }

            template <typename TR>
            NDArray<TR>& max(const Axes &axes, NDArray<TR> &out) const noexcept {
                const NDArray<T> reorder = this->axes_reorder(axes);

                NDIter out_iter(out.data_, out);
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T max;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    max = src_iter.next();
                    
                    for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                        max = laruen::math::common::max(max, src_iter.next());
                    }
                    out_iter.next() = max;
                }

                return out;
            }

            template <typename TR = T>
            NDArray<TR> max(const Axes &axes) const noexcept {
                NDArray<TR> out(*this, laruen::multi::utils::compress_axes(axes, this->ndim_));
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
                NDArray<uint_fast64_t> out(*this, laruen::multi::utils::compress_axes(axes, this->ndim_));
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
                NDIter src_iter(reorder.data_, reorder);
                uint_fast64_t sample_size = reorder.size_ / out.size_;
                T min;

                for(uint_fast64_t i = 0;i < out.size_;i++) {
                    min = src_iter.next();
                    
                    for(uint_fast64_t j = 0;j < sample_size - 1;j++) {
                        min = laruen::math::common::min(min, src_iter.next());
                    }
                    out_iter.next() = min;
                }

                return out;
            }
            
            template <typename TR = T>
            NDArray<TR> min(const Axes &axes) const noexcept {
                NDArray<TR> out(*this, laruen::multi::utils::compress_axes(axes, this->ndim_));
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
                NDArray<uint_fast64_t> out(*this, laruen::multi::utils::compress_axes(axes, this->ndim_));
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
            template <typename TT>
            bool operator==(const NDArray<TT> &ndarray) const noexcept {
                bool eq = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && eq;i++) {
                    eq = (lhs_iter.next() == rhs_iter.next());
                }

                return eq;
            }
            
            template <typename TT>
            bool operator!=(const NDArray<TT> &ndarray) const noexcept {
                return !(*this == ndarray);
            }
            
            template <typename TT>
            bool operator>=(const NDArray<TT> &ndarray) const noexcept {
                bool ge = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && ge;i++) {
                    ge = (lhs_iter.next() >= rhs_iter.next());
                }

                return ge;
            }
            
            template <typename TT>
            bool operator<=(const NDArray<TT> &ndarray) const noexcept {
                bool le = this->shape_ == ndarray.shape_;
                NDIter lhs_iter(this->data_, *this);
                NDIter rhs_iter(ndarray.data_, ndarray);

                for(uint_fast64_t i = 0;i < this->size_ && le;i++) {
                    le = (lhs_iter.next() <= rhs_iter.next());
                }

                return le;
            }
            
            template <typename TT>
            bool operator>(const NDArray<TT> &ndarray) const noexcept {
                return !(*this <= ndarray);
            }
            
            template <typename TT>
            bool operator<(const NDArray<TT> &ndarray) const noexcept {
                return !(*this >= ndarray);
            }

            NDArray<T> transpose(uint_fast8_t dim_begin, uint_fast8_t dim_end) {
                NDArray<T> result = this->view();
                result.contig_ = false;

                const uint_fast8_t middle = (dim_end + dim_begin) >> 1;

                for(;dim_begin < middle;dim_begin++) {
                    dim_end--;

                    result.shape_[dim_begin] = this->shape_[dim_end];
                    result.shape_[dim_end] = this->shape_[dim_begin];
                    result.strides_[dim_begin] = this->strides_[dim_end];
                    result.strides_[dim_end] = this->strides_[dim_begin];
                    result.dim_sizes_[dim_begin] = this->dim_sizes_[dim_end];
                    result.dim_sizes_[dim_end] = this->dim_sizes_[dim_begin];
                }

                return result;
            }

            const NDArray<T> transpose(uint_fast8_t dim_begin, uint_fast8_t dim_end) const {
                NDArray<T> result = this->view();
                result.contig_ = false;

                const uint_fast8_t middle = (dim_end + dim_begin) >> 1;

                for(;dim_begin < middle;dim_begin++) {
                    dim_end--;

                    result.shape_[dim_begin] = this->shape_[dim_end];
                    result.shape_[dim_end] = this->shape_[dim_begin];
                    result.strides_[dim_begin] = this->strides_[dim_end];
                    result.strides_[dim_end] = this->strides_[dim_begin];
                    result.dim_sizes_[dim_begin] = this->dim_sizes_[dim_end];
                    result.dim_sizes_[dim_end] = this->dim_sizes_[dim_begin];
                }

                return result;
            }

            inline NDArray<T> transpose() noexcept {
                return this->transpose(0, this->ndim_);
            }

            inline const NDArray<T> transpose() const noexcept {
                return this->transpose(0, this->ndim_);
            }

            NDArray<T> view_reshape(const Shape &shape) noexcept {
                assert(this->contig_);
                NDArray<T> view = this->view();
                view.reshape(shape);
                return view;
            }

            const NDArray<T> view_reshape(const Shape &shape) const noexcept {
                assert(this->contig_);
                NDArray<T> view = this->view();
                view.reshape(shape);
                return view;
            }

            NDArray<T> copy_reshape(const Shape &shape) const noexcept {
                NDArray<T> copy(shape);
                assert(copy.size_ == this->size_);
                copy.copy_data_from(*this);
                return copy;
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
            const NDArray<T> expansion(const ArrayBase &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expansion */
                
                NDArray<T> expansion(this->data_, Shape(expand_to.shape_), Strides(expand_to.ndim_, 0),
                Strides(expand_to.ndim_, 0), expand_to.size_, expand_to.ndim_, false, false);
                
                uint_fast8_t expansion_idx = expansion.ndim_ - this->ndim_;

                for(uint_fast8_t expanded_idx = 0;expanded_idx < this->ndim_;expanded_idx++, expansion_idx++) {
                    if(expansion.shape_[expansion_idx] == this->shape_[expanded_idx]) {
                        expansion.strides_[expansion_idx] = this->strides_[expanded_idx];
                        expansion.dim_sizes_[expansion_idx] = this->dim_sizes_[expanded_idx];
                    }
                }

                return expansion;
            }

            const NDArray<T> matmul_expansion(const ArrayBase &expand_to) const noexcept {
                /* expand the dimensions of this to the dimensions of expand_to */
                
                NDArray<T> expansion(this->data_, Shape(expand_to.shape_), Strides(expand_to.ndim_, 0),
                Strides(expand_to.ndim_, 0), expand_to.size_, expand_to.ndim_, false, false);
                
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
                Strides(this->ndim_), this->size_, this->ndim_, false, false);

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
            // getters & setters
            inline void data(T *data) noexcept {
                this->data_ = data;
            }

            inline void data(const T *data) const noexcept {
                /*
                *** important ***
                This function is used to change the address
                the pointer this->data_ points to without
                having access to the data.
                Use only when 'data' is not actually const!
                else could lead to undefined behavior
                */
                static_assert(!std::is_const_v<T>);

                this->data_ = const_cast<T*>(data);
            }

            inline const T* data() const noexcept {
                return this->data_;
            }

            inline T* data() noexcept {
                return this->data_;
            }

            inline bool& data_owner() const noexcept {
                return this->data_owner_;
            }

            inline T& operator[](uint_fast64_t index) noexcept {
                return this->data_[index];
            }

            inline const T& operator[](uint_fast64_t index) const noexcept {
                return this->data_[index];
            }

            // arithmetical functions
            inline NDArray& negate_eq() const noexcept {
                Impl::negate_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& negate(NDArray<TR> &out) const {
                Impl::negate(this->data_, *this, out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> negate() const noexcept {
                NDArray<TR> out(this->shape_);
                this->negate(out);
                return out;
            }

            template <typename TT>
            inline NDArray& add_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::add_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& add_eq(T value) noexcept {
                Impl::add_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& add(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::add(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& add(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::add(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> add(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->add(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> add(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->add(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> add(const NDArray<TT> &rhs) const noexcept {
                return this->template add<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& subtract_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::subtract_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& subtract_eq(T value) noexcept {
                Impl::subtract_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& subtract(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;
                
                Impl::subtract(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& subtract(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::subtract(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> subtract(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->subtract(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> subtract(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->subtract(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> subtract(const NDArray<TT> &rhs) const noexcept {
                return this->template subtract<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& multiply_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::multiply_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& multiply_eq(T value) noexcept {
                Impl::multiply_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& multiply(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::multiply(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& multiply(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::multiply(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> multiply(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->multiply(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> multiply(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->multiply(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> multiply(const NDArray<TT> &rhs) const noexcept {
                return this->template multiply<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& divide_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::divide_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& divide_eq(T value) noexcept {
                Impl::divide_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& divide(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::divide(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& divide(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::divide(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> divide(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->divide(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> divide(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->divide(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> divide(const NDArray<TT> &rhs) const noexcept {
                return this->template divide<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& bit_xor_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::bit_xor_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& bit_xor_eq(T value) noexcept {
                Impl::bit_xor_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& bit_xor(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::bit_xor(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_xor(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::bit_xor(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> bit_xor(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_xor(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_xor(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->bit_xor(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> bit_xor(const NDArray<TT> &rhs) const noexcept {
                return this->template bit_xor<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& bit_and_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::bit_and_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& bit_and_eq(T value) noexcept {
                Impl::bit_and_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& bit_and(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::bit_and(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_and(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::bit_and(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> bit_and(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_and(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_and(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->bit_and(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> bit_and(const NDArray<TT> &rhs) const noexcept {
                return this->template bit_and<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& bit_or_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::bit_or_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& bit_or_eq(T value) noexcept {
                Impl::bit_or_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& bit_or(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::bit_or(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_or(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::bit_or(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> bit_or(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->bit_or(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> bit_or(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->bit_or(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> bit_or(const NDArray<TT> &rhs) const noexcept {
                return this->template bit_or<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& shl_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::shl_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& shl_eq(T value) noexcept {
                Impl::shl_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& shl(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::shl(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shl(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::shl(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> shl(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->shl(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shl(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->shl(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> shl(const NDArray<TT> &rhs) const noexcept {
                return this->template shl<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& shr_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::shr_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& shr_eq(T value) noexcept {
                Impl::shr_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& shr(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::shr(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shr(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::shr(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> shr(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->shr(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> shr(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->shr(value, out);
                return out;
            }

            inline NDArray& bit_not_eq() noexcept {
                Impl::bit_not_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& bit_not(NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::bit_not(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> bit_not() const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->bit_not(out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> shr(const NDArray<TT> &rhs) const noexcept {
                return this->template shr<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& remainder_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::remainder_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& remainder_eq(T value) noexcept {
                Impl::remainder_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& remainder(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::remainder(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& remainder(TR value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::remainder(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> remainder(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->remainder(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> remainder(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->remainder(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> remainder(const NDArray<TT> &rhs) const noexcept {
                return this->template remainder<types::result_type_t<T, TT>, TT>(rhs);
            }

            template <typename TT>
            inline NDArray& power_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::power_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            template <typename TT>
            inline NDArray& power_eq(TT value) noexcept {
                Impl::power_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& power(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::power(this->data_, this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                rhs.data_, rhs.shape_ == out.shape_ ? rhs : (rhs_exp = rhs.expansion(out)),
                out.data_, out);
                return out;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& power(TT value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::power(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> power(const NDArray<TT> &rhs) const noexcept {
                NDArray<TR> out(laruen::multi::utils::broadcast(this->shape_, rhs.shape_));
                this->power(rhs, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> power(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->power(value, out);
                return out;
            }

            template <typename TR, typename TT, typename = std::enable_if_t<!std::is_same_v<TR, TT>>>
            inline NDArray<TR> power(TT value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->power(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> power(const NDArray<TT> &rhs) const noexcept {
                return this->template power<types::result_type_t<T, TT>, TT>(rhs);
            }

            inline NDArray& ln_eq() noexcept {
                Impl::ln_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& ln(NDArray<TR> &out) const {
                Impl::ln(this->data_, *this, out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> ln() const noexcept {
                NDArray<TR> out(this->shape_);
                this->ln(out);
                return out;
            }

            inline NDArray& log2_eq() noexcept {
                Impl::log2_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& log2(NDArray<TR> &out) const {
                Impl::log2(this->data_, *this, out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> log2() const noexcept {
                NDArray<TR> out(this->shape_);
                this->log2(out);
                return out;
            }

            inline NDArray& log10_eq() noexcept {
                Impl::log10_eq(this->data_, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& log10(NDArray<TR> &out) const {
                Impl::log10(this->data_, *this, out.data_, out);
                return out;
            }

            template <typename TR = T>
            inline NDArray<TR> log10() const noexcept {
                NDArray<TR> out(this->shape_);
                this->log10(out);
                return out;
            }

            template <typename TT>
            inline NDArray& log_eq(const NDArray<TT> &base) {
                Impl::log_eq(this->data_, *this, base.data_, base);
                return *this;
            }

            template <typename TT>
            inline NDArray& log_eq(TT base) noexcept {
                Impl::log_eq(this->data_, *this, base);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& log(const NDArray<TT> &base, NDArray<TR> &out) const {
                Impl::log(this->data_, *this, base.data_, base, out.data_, out);
                return out;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& log(TT base, NDArray<TR> &out) const {
                Impl::log(this->data_, *this, base, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> log(const NDArray<TT> &base) const {
                NDArray<TR> out(this->shape_);
                this->log(base, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> log(TT base) const {
                NDArray<TR> out(this->shape_);
                this->log(base, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> log(const NDArray<TT> &base) const {
                NDArray<types::result_type_t<T, TT>> out(this->shape_);
                this->log(base, out);
                return out;
            }

            template <typename TT, typename = std::enable_if_t<!types::is_ndarray_v<TT>>>
            inline NDArray<types::result_type_t<T, TT>> log(TT base) const {
                NDArray<types::result_type_t<T, TT>> out(this->shape_);
                this->log(base, out);
                return out;
            }

            // inverse math functions
            template <typename TT>
            inline NDArray& inverse_subtract_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::inverse_subtract_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& inverse_subtract_eq(T value) noexcept {
                Impl::inverse_subtract_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& inverse_subtract(T value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::inverse_subtract(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_subtract(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_subtract(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray& inverse_divide_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::inverse_divide_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& inverse_divide_eq(T value) noexcept {
                Impl::inverse_divide_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& inverse_divide(T value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::inverse_divide(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_divide(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_divide(value, out);
                return out;
            }
            
            template <typename TT>
            inline NDArray& inverse_shl_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::inverse_shl_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& inverse_shl_eq(T value) noexcept {
                Impl::inverse_shl_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& inverse_shl(T value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::inverse_shl(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_shl(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_shl(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray& inverse_shr_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;
                
                Impl::inverse_shr_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& inverse_shr_eq(T value) noexcept {
                Impl::inverse_shr_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& inverse_shr(T value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::inverse_shr(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_shr(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_shr(value, out);
                return out;
            }
            
            template <typename TT>
            inline NDArray& inverse_remainder_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::inverse_remainder_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            inline NDArray& inverse_remainder_eq(T value) noexcept {
                Impl::inverse_remainder_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& inverse_remainder(T value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;

                Impl::inverse_remainder(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_remainder(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_remainder(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray& inverse_power_eq(const NDArray<TT> &rhs) noexcept {
                const NDArray<TT> rhs_exp;

                Impl::inverse_power_eq(this->data_, *this, rhs.data_,
                this->shape_ == rhs.shape_ ? rhs : (rhs_exp = rhs.expansion(*this)));
                return *this;
            }

            template <typename TT>
            inline NDArray& inverse_power_eq(TT value) noexcept {
                Impl::inverse_power_eq(this->data_, *this, value);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& inverse_power(TT value, NDArray<TR> &out) const noexcept {
                const NDArray<T> lhs_exp;
                
                Impl::inverse_power(this->data_,
                this->shape_ == out.shape_ ? *this : (lhs_exp = this->expansion(out)),
                value, out.data_, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR> inverse_power(TR value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_power(value, out);
                return out;
            }

            template <typename TR, typename TT, typename = std::enable_if_t<!std::is_same_v<TR, TT>>>
            inline NDArray<TR> inverse_power(TT value) const noexcept {
                NDArray<TR> out(new TR[this->size_], *this, true);
                this->inverse_power(value, out);
                return out;
            }

            template <typename TT>
            inline NDArray& inverse_log_eq(const NDArray<TT> &power) {
                Impl::inverse_log_eq(this->data_, *this, power.data_, power);
                return *this;
            }
            
            template <typename TT>
            inline NDArray& inverse_log_eq(TT power) noexcept {
                Impl::inverse_log_eq(this->data_, *this, power);
                return *this;
            }

            template <typename TT, typename TR>
            inline NDArray<TR>& inverse_log(TT power, NDArray<TR> &out) const {
                Impl::inverse_log(this->data_, *this, power, out.data_, out);
                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> inverse_log(TT power) const {
                NDArray<TR> out(this->shape_);
                Impl::inverse_log(this->data_, *this, power, out.data_, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> inverse_log(TT power) const {
                NDArray<types::result_type_t<T, TT>> out(this->shape_);
                Impl::inverse_log(this->data_, *this, power, out.data_, out);
                return out;
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

            template <typename TT, typename TR>
            inline NDArray<TR>& matmul(const NDArray<TT> &rhs, NDArray<TR> &out) const noexcept {
                using laruen::math::common::min, laruen::math::common::is_pow2;
                using laruen::math::bits::lsb64;

                uint_fast64_t lhs_rows = this->shape_[this->ndim_ - 2];
                uint_fast64_t rhs_cols = rhs.shape_.back();
                uint_fast64_t lhs_shared = this->shape_.back();

                assert(lhs_rows == out.shape_[out.ndim_ - 2] &&
                rhs_cols == out.shape_.back() &&
                lhs_shared == rhs.shape_[rhs.ndim_ - 2]);

                uint_fast8_t depth = min(lsb64(lhs_rows), min(lsb64(rhs_cols), lsb64(lhs_shared)));

                if(depth && is_pow2(lhs_rows) && is_pow2(rhs_cols) && is_pow2(lhs_shared)) {
                    depth--;
                }

                const NDArray<T> lhs_exp;
                const NDArray<TT> rhs_exp;

                Impl::matmul(this->data_,
                std::equal(out.shape_.cbegin(), out.shape_.cend() - 2, this->shape_.cbegin())
                ? *this : (lhs_exp = this->matmul_expansion(out)),
                rhs.data_,
                std::equal(out.shape_.cbegin(), out.shape_.cend() - 2, rhs.shape_.cbegin())
                ? rhs : (rhs_exp = rhs.matmul_expansion(out)),
                out.data_, out, depth);

                return out;
            }

            template <typename TR, typename TT>
            inline NDArray<TR> matmul(const NDArray<TT> &rhs) const {
                NDArray<TR> out(laruen::multi::utils::matmul_broadcast(this->shape_, rhs.shape_));
                this->matmul(rhs, out);
                return out;
            }

            template <typename TT>
            inline NDArray<types::result_type_t<T, TT>> matmul(const NDArray<TT> &rhs) const {
                return this->matmul<types::result_type_t<T, TT>, TT>(rhs);
            }

            // arithmetical operators
            template <typename TT>
            inline NDArray& operator+=(const NDArray<TT> &rhs) {
                return this->add_eq(rhs);
            }

            inline NDArray& operator+=(T value) noexcept {
                return this->add_eq(value);
            }

            template <typename TT>
            inline auto operator+(const NDArray<TT> &rhs) const {
                return this->add(rhs);
            }

            inline NDArray<T> operator+(T value) const noexcept {
                return this->add(value);
            }

            template <typename TT>
            inline NDArray& operator-=(const NDArray<TT> &rhs) {
                return this->subtract_eq(rhs);
            }

            inline NDArray& operator-=(T value) noexcept {
                return this->subtract_eq(value);
            }
            
            template <typename TT>
            inline auto operator-(const NDArray<TT> &rhs) const {
                return this->subtract(rhs);
            }

            inline NDArray<T> operator-(T value) const noexcept {
                return this->subtract(value);
            }

            inline NDArray<T> operator-() const noexcept {
                return this->negate();
            }

            template <typename TT>
            inline NDArray& operator*=(const NDArray<TT> &rhs) {
                return this->multiply_eq(rhs);
            }

            inline NDArray& operator*=(T value) noexcept {
                return this->multiply_eq(value);
            }

            template <typename TT>
            inline auto operator*(const NDArray<TT> &rhs) const {
                return this->multiply(rhs);
            }

            inline NDArray<T> operator*(T value) const noexcept {
                return this->multiply(value);
            }

            template <typename TT>
            inline NDArray& operator/=(const NDArray<TT> &rhs) {
                return this->divide_eq(rhs);
            }

            inline NDArray& operator/=(T value) noexcept {
                return this->divide_eq(value);
            }

            template <typename TT>
            inline auto operator/(const NDArray<TT> &rhs) const {
                return this->divide(rhs);
            }

            inline NDArray<T> operator/(T value) const noexcept {
                return this->divide(value);
            }
            
            template <typename TT>
            inline NDArray& operator^=(const NDArray<TT> &rhs) {
                return this->bit_xor_eq(rhs);
            }

            inline NDArray& operator^=(T value) noexcept {
                return this->bit_xor_eq(value);
            }

            template <typename TT>
            inline auto operator^(const NDArray<TT> &rhs) const {
                return this->bit_xor(rhs);
            }

            inline NDArray<T> operator^(T value) const noexcept {
                return this->bit_xor(value);
            }
            
            template <typename TT>
            inline NDArray& operator&=(const NDArray<TT> &rhs) {
                return this->bit_and_eq(rhs);
            }

            inline NDArray& operator&=(T value) noexcept {
                return this->bit_and_eq(value);
            }

            template <typename TT>
            inline auto operator&(const NDArray<TT> &rhs) const {
                return this->bit_and(rhs);
            }

            inline NDArray<T> operator&(T value) const noexcept {
                return this->bit_and(value);
            }

            template <typename TT>
            inline NDArray& operator|=(const NDArray<TT> &rhs) {
                return this->bit_or_eq(rhs);
            }
            
            inline NDArray& operator|=(T value) noexcept {
                return this->bit_or_eq(value);
            }

            template <typename TT>
            inline auto operator|(const NDArray<TT> &rhs) const {
                return this->bit_or(rhs);
            }

            inline NDArray<T> operator|(T value) const noexcept {
                return this->bit_or(value);
            }
            
            template <typename TT>
            inline NDArray& operator<<=(const NDArray<TT> &rhs) {
                return this->shl_eq(rhs);
            }

            inline NDArray& operator<<=(T value) noexcept {
                return this->shl_eq(value);
            }
            template <typename TT>

            inline auto operator<<(const NDArray<TT> &rhs) const {
                return this->shl(rhs);
            }

            inline NDArray<T> operator<<(T value) const noexcept {
                return this->shl(value);
            }
            
            template <typename TT>
            inline NDArray& operator>>=(const NDArray<TT> &rhs) {
                return this->shr_eq(rhs);
            }

            inline NDArray& operator>>=(T value) noexcept {
                return this->shr_eq(value);
            }

            template <typename TT>
            inline auto operator>>(const NDArray<TT> &rhs) const {
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

            template <typename TT>
            inline NDArray& operator%=(const NDArray<TT> &rhs) {
                return this->remainder_eq(rhs);
            }

            template <typename TT>
            inline auto operator%(const NDArray<TT> &rhs) const {
                return this->remainder(rhs);
            }

            inline NDArray<T> operator%(T value) const noexcept {
                return this->remainder(value);
            }

            // friend operators
            inline friend NDArray<T> operator+(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray + value;
            }

            inline friend NDArray<T> operator-(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.inverse_subtract(value);
            }

            inline friend NDArray<T> operator*(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray * value;
            }
            
            inline friend NDArray<T> operator/(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.inverse_divide(value);
            }
            
            inline friend NDArray<T> operator^(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.bit_xor(value);
            }

            inline friend NDArray<T> operator&(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.bit_and(value);
            }

            inline friend NDArray<T> operator|(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.bit_or(value);
            }

            inline friend NDArray<T> operator<<(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.inverse_shl(value);
            }

            inline friend NDArray<T> operator>>(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.inverse_shr(value);
            }

            inline friend NDArray<T> operator%(T value, const NDArray<T> &ndarray) noexcept {
                return ndarray.inverse_remainder(value);
            }

            // more utility functions
            inline ArrayBase& arraybase() noexcept {
                return *this;
            }

            inline const ArrayBase& arraybase() const noexcept {
                return *this;
            }
            
            inline NDArray<T> view() noexcept {
                return NDArray<T>(this->data_, this->shape_, this->strides_,
                this->dim_sizes_, this->size_, this->ndim_, this->contig_, false);
            }

            inline const NDArray<T> view() const noexcept {
                return NDArray<T>(this->data_, this->shape_, this->strides_,
                this->dim_sizes_, this->size_, this->ndim_, this->contig_, false);
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const NDArray &ndarray) noexcept {
                return stream << ndarray.str();
            }
    };
};

#endif