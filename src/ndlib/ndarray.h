
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
#include "src/ndlib/utils.h"
#include "src/ndlib/types.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/array_base.h"
#include "src/ndlib/range.h"
#include "src/ndlib/type_selection.h"
#include "src/ndlib/impl.h"


namespace laruen::ndlib {

    template <typename T = float64_t>
    class NDArray : public ArrayBase {

        template <typename> friend class NDArray;

        private:
            T *m_data;
            const NDArray<T> *m_base = nullptr;

        public:
            typedef T DType;

            ~NDArray();

            // constructors and assignment operators
            NDArray() noexcept;

            NDArray(std::initializer_list<T> init_list) noexcept;

            NDArray(std::initializer_list<T> init_list, const Shape &shape) noexcept;

            NDArray(T *data, const Shape &shape, const Strides &strides, const Strides &dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, const NDArray<T> *base = nullptr) noexcept;

            NDArray(T *data, Shape &&shape, Strides &&strides, Strides &&dim_sizes,
            uint_fast64_t size, uint_fast8_t ndim, const NDArray<T> *base = nullptr) noexcept;
            
            explicit NDArray(const Shape &shape) noexcept;
            
            NDArray(const Shape &shape, T value) noexcept;
            
            NDArray(T *data, const ArrayBase &arraybase, const NDArray<T> *base = nullptr) noexcept;
            
            NDArray(const NDArray &ndarray) noexcept;
            
            NDArray(NDArray &&ndarray) noexcept;
            
            explicit NDArray(const Range<T> &range) noexcept;
            
            NDArray(const Range<T> &range, const Shape &shape);
            
            NDArray(const ArrayBase &arraybase, const Axes &axes) noexcept;
            
            NDArray(NDArray<T> &ndarray, const SliceRanges &ranges) noexcept;
            
            template <typename T2>
            NDArray(const NDArray<T2> &ndarray) noexcept;
            
            template <typename T2>
            NDArray(NDArray<T2> &&ndarray) noexcept;

            NDArray& operator=(const NDArray &ndarray) noexcept;
            
            NDArray& operator=(NDArray &&ndarray) noexcept;
            
            template <typename T2>
            NDArray& operator=(const NDArray<T2> &ndarray) noexcept;
            
            template <typename T2>
            NDArray& operator=(NDArray<T2> &&ndarray) noexcept;

            // utility functions
            template <typename T2>
            void copy_data_from(const NDArray<T2> &ndarray) noexcept;
            
            void fill(T value) noexcept;

            // computational functions on the array
            template <typename TR>
            NDArray<TR>& sum(const Axes &axes, NDArray<TR> &out) const noexcept;

            NDArray<T> sum(const Axes &axes) const noexcept;

            T sum() const noexcept;

            template <typename TR>
            NDArray<TR>& max(const Axes &axes, NDArray<TR> &out) const noexcept;

            NDArray<T> max(const Axes &axes) const noexcept;
            
            T max() const noexcept;
            
            NDArray<uint_fast64_t>& indices_max(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept;

            NDArray<uint_fast64_t> indices_max(const Axes &axes) const noexcept;

            uint_fast64_t index_max() const noexcept;
            
            NDIndex ndindex_max() const noexcept;

            template <typename TR>
            NDArray<TR>& min(const Axes &axes, NDArray<TR> &out) const noexcept;
            
            NDArray<T> min(const Axes &axes) const noexcept;

            T min() const noexcept;
            
            NDArray<uint_fast64_t>& indices_min(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept;

            NDArray<uint_fast64_t> indices_min(const Axes &axes) const noexcept;

            uint_fast64_t index_min() const noexcept;
            
            NDIndex ndindex_min() const noexcept;

            // indexing and slicing operators
            T& operator[](const NDIndex &ndindex) noexcept;

            const T& operator[](const NDIndex &ndindex) const noexcept;

            NDArray<T> operator[](const SliceRanges &ranges) noexcept;
            
            const NDArray<T> operator[](const SliceRanges &ranges) const noexcept;

            // bool operators between arrays
            template <typename T2>
            bool operator==(const NDArray<T2> &ndarray) const noexcept;
            
            template <typename T2>
            bool operator!=(const NDArray<T2> &ndarray) const noexcept;
            
            template <typename T2>
            bool operator>=(const NDArray<T2> &ndarray) const noexcept;
            
            template <typename T2>
            bool operator<=(const NDArray<T2> &ndarray) const noexcept;
            
            template <typename T2>
            bool operator>(const NDArray<T2> &ndarray) const noexcept;
            
            template <typename T2>
            bool operator<(const NDArray<T2> &ndarray) const noexcept;

            NDArray<T> transpose() noexcept;

            const NDArray<T> transpose() const noexcept;

            // string function
            std::string str() const noexcept;

        private:
            // private utility functions
            template <typename T2>
            const NDArray<T> expand_to(const NDArray<T2> &expand_to) const noexcept;

            const NDArray<T> axes_reorder(const Axes &axes) const noexcept;

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
                impl::add_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& add_eq(T value) noexcept {
                impl::add_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& add(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::add(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& add(TR value, NDArray<TR> &out) const noexcept {
                impl::add(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::subtract_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& subtract_eq(T value) noexcept {
                impl::subtract_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& subtract(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::subtract(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& subtract(TR value, NDArray<TR> &out) const noexcept {
                impl::subtract(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::multiply_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& multiply_eq(T value) noexcept {
                impl::multiply_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& multiply(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::multiply(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& multiply(TR value, NDArray<TR> &out) const noexcept {
                impl::multiply(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::divide_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& divide_eq(T value) noexcept {
                impl::divide_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& divide(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::divide(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& divide(TR value, NDArray<TR> &out) const noexcept {
                impl::divide(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::bit_xor_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& bit_xor_eq(T value) noexcept {
                impl::bit_xor_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_xor(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::bit_xor(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_xor(TR value, NDArray<TR> &out) const noexcept {
                impl::bit_xor(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::bit_and_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& bit_and_eq(T value) noexcept {
                impl::bit_and_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_and(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::bit_and(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_and(TR value, NDArray<TR> &out) const noexcept {
                impl::bit_and(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::bit_or_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& bit_or_eq(T value) noexcept {
                impl::bit_or_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& bit_or(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::bit_or(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& bit_or(TR value, NDArray<TR> &out) const noexcept {
                impl::bit_or(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::shl_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& shl_eq(T value) noexcept {
                impl::shl_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shl(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::shl(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shl(TR value, NDArray<TR> &out) const noexcept {
                impl::shl(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::shr_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& shr_eq(T value) noexcept {
                impl::shr_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& shr(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::shr(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& shr(TR value, NDArray<TR> &out) const noexcept {
                impl::shr(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::bit_not_eq(this->m_data, *this);
                return *this;
            }

            template <typename TR>
            inline NDArray<TR>& bit_not(NDArray<TR> &out) const noexcept {
                impl::bit_not(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::remainder_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& remainder_eq(T value) noexcept {
                impl::remainder_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& remainder(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::remainder(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& remainder(TR value, NDArray<TR> &out) const noexcept {
                impl::remainder(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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
                impl::power_eq(this->m_data, *this, rhs.m_data,
                this->m_shape == rhs.m_shape ? rhs : rhs.expand_to(*this));
                return *this;
            }

            inline NDArray& power_eq(T value) noexcept {
                impl::power_eq(this->m_data, *this, value);
                return *this;
            }

            template <typename T2, typename TR>
            inline NDArray<TR>& power(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept {
                impl::power(this->m_data, this->m_shape == out.m_shape ? *this : this->expand_to(out),
                rhs.m_data, rhs.m_shape == out.m_shape ? rhs : rhs.expand_to(out),
                out.m_data, out);
                return out;
            }

            template <typename TR>
            inline NDArray<TR>& power(TR value, NDArray<TR> &out) const noexcept {
                impl::power(this->m_data,
                this->m_shape == out.m_shape ? *this : this->expand_to(out),
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

#include "src/ndlib/ndarray.tpp"
#endif