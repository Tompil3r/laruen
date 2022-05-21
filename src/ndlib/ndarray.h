
#ifndef NDARRAY_H
#define NDARRAY_H

#include "src/ndlib/ndlib_utils.h"
#include "src/ndlib/ndlib_types.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/array_base.h"
#include "src/utils/range.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <type_traits>
#include <ostream>

using laruen::utils::Range;

namespace laruen::ndlib {

    template <typename T = float64_t, bool C = true>
    class NDArray : public ArrayBase {

        template <typename, bool> friend class NDArray;
        friend class NDIter<NDArray<T, C>, C>;
        friend class NDIter<const NDArray<T, C>, C>;

        private:
            T *m_data;

        public:
            typedef ArrayBase Base;
            typedef T DType;
            static constexpr bool CONTIGUOUS = C;

            ~NDArray();

            // constructors and assignment operators
            NDArray() noexcept;

            NDArray(T *data, const Shape &shape, const Strides &strides,
            uint_fast64_t size, uint_fast8_t ndim, bool free_mem) noexcept;

            NDArray(T *data, Shape &&shape, Strides &&strides,
            uint_fast64_t size, uint_fast8_t ndim, bool free_mem) noexcept;
            
            NDArray(const Shape &shape) noexcept;
            
            NDArray(const Shape &shape, T value) noexcept;
            
            NDArray(T *data, const ArrayBase &base) noexcept;
            
            NDArray(T *data, const ArrayBase &base, bool free_mem) noexcept;
            
            NDArray(const NDArray &ndarray) noexcept;
            
            NDArray(NDArray &&ndarray) noexcept;
            
            NDArray(const Range<T> &range) noexcept;
            
            NDArray(const Range<T> &range, const Shape &shape);
            
            NDArray(const ArrayBase &base, const Axes &axes) noexcept;
            
            template <bool C2>
            NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges) noexcept;
            
            template <typename T2, bool C2>
            NDArray(const NDArray<T2, C2> &ndarray) noexcept;
            
            template <typename T2, bool C2>
            NDArray(NDArray<T2, C2> &&ndarray) noexcept;

            NDArray& operator=(const NDArray &ndarray) noexcept;
            
            NDArray& operator=(NDArray &&ndarray) noexcept;
            
            template <typename T2, bool C2>
            NDArray& operator=(const NDArray<T2, C2> &ndarray) noexcept;
            
            template <typename T2, bool C2>
            NDArray& operator=(NDArray<T2, C2> &&ndarray) noexcept;

            // utility functions
            template <typename T2, bool C2>
            void copy_data_from(const NDArray<T2, C2> &ndarray) noexcept;
            
            void fill(T value) noexcept;

            // computational functions on the array
            template <typename TR, bool CR>
            NDArray<TR, CR>& sum(const Axes &axes, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> sum(const Axes &axes) const noexcept;

            T sum() const noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& max(const Axes &axes, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> max(const Axes &axes) const noexcept;
            
            T max() const noexcept;
            
            uint_fast64_t index_max() const noexcept;
            
            NDIndex ndindex_max() const noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& min(const Axes &axes, NDArray<TR, CR> &out) const noexcept;
            
            NDArray<T, true> min(const Axes &axes) const noexcept;

            T min() const noexcept;
            
            uint_fast64_t index_min() const noexcept;
            
            NDIndex ndindex_min() const noexcept;

            // indexing and slicing operators
            T& operator[](const NDIndex &ndindex) noexcept;

            const T& operator[](const NDIndex &ndindex) const noexcept;

            NDArray<T, false> operator[](const SliceRanges &ranges) noexcept;
            
            const NDArray<T, false> operator[](const SliceRanges &ranges) const noexcept;

            // arithmetical functions
            template <typename T2, bool C2>
            NDArray& add_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& add_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& add_eq(const NDArray<T2, C2> &rhs);

            NDArray& add_eq(T value) noexcept;
            
            template <typename TR, bool CR>
            NDArray<TR, CR>& add(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> add(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& add_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> add_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& add_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> add_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& add(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> add(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& subtract_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& subtract_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& subtract_eq(const NDArray<T2, C2> &rhs);

            NDArray& subtract_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& subtract(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> subtract(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& subtract_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> subtract_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& subtract_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> subtract_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& subtract(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> subtract(const NDArray<T2, C2> &rhs) const;
            
            template <typename T2, bool C2>
            NDArray& multiply_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& multiply_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& multiply_eq(const NDArray<T2, C2> &rhs);

            NDArray& multiply_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& multiply(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> multiply(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& multiply_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> multiply_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& multiply_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> multiply_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& multiply(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> multiply(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& divide_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& divide_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& divide_eq(const NDArray<T2, C2> &rhs);

            NDArray& divide_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& divide(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> divide(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& divide_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> divide_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& divide_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> divide_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& divide(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> divide(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& bit_xor_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& bit_xor_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& bit_xor_eq(const NDArray<T2, C2> &rhs);

            NDArray& bit_xor_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& bit_xor(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> bit_xor(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_xor_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_xor_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_xor_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_xor_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_xor(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_xor(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& bit_and_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& bit_and_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& bit_and_eq(const NDArray<T2, C2> &rhs);

            NDArray& bit_and_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& bit_and(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> bit_and(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_and_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_and_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_and_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_and_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_and(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_and(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& bit_or_eq_b(const NDArray<T2, C2> &rhs);
            
            template <typename T2, bool C2>
            NDArray& bit_or_eq_r(const NDArray<T2, C2> &rhs);
            
            template <typename T2, bool C2>
            NDArray& bit_or_eq(const NDArray<T2, C2> &rhs);

            NDArray& bit_or_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& bit_or(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> bit_or(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_or_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_or_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_or_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_or_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& bit_or(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> bit_or(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& shl_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& shl_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& shl_eq(const NDArray<T2, C2> &rhs);

            NDArray& shl_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& shl(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> shl(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shl_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shl_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shl_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shl_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shl(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shl(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& shr_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& shr_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& shr_eq(const NDArray<T2, C2> &rhs);

            NDArray& shr_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& shr(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> shr(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shr_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shr_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shr_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shr_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& shr(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> shr(const NDArray<T2, C2> &rhs) const;

            NDArray& bit_not_eq() noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& bit_not(NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> bit_not() const noexcept;

            template <typename T2, bool C2>
            NDArray& remainder_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& remainder_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& remainder_eq(const NDArray<T2, C2> &rhs);

            NDArray& remainder_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& remainder(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> remainder(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& remainder_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> remainder_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& remainder_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> remainder_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& remainder(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> remainder(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2>
            NDArray& power_eq_b(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& power_eq_r(const NDArray<T2, C2> &rhs);

            template <typename T2, bool C2>
            NDArray& power_eq(const NDArray<T2, C2> &rhs);

            NDArray& power_eq(T value) noexcept;

            template <typename TR, bool CR>
            NDArray<TR, CR>& power(T value, NDArray<TR, CR> &out) const noexcept;

            NDArray<T, true> power(T value) const noexcept;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& power_b(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> power_b(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& power_r(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> power_r(const NDArray<T2, C2> &rhs) const;

            template <typename T2, bool C2, typename TR, bool CR>
            NDArray<TR, CR>& power(const NDArray<T2, C2> &rhs, NDArray<TR, CR> &out) const;

            template <typename T2, bool C2>
            NDArray<types::combine_types_t<T, T2>, true> power(const NDArray<T2, C2> &rhs) const;

            // bool operators between arrays
            template <typename T2, bool C2>
            bool operator==(const NDArray<T2, C2> &ndarray) const noexcept;
            
            template <typename T2, bool C2>
            bool operator!=(const NDArray<T2, C2> &ndarray) const noexcept;
            
            template <typename T2, bool C2>
            bool operator>=(const NDArray<T2, C2> &ndarray) const noexcept;
            
            template <typename T2, bool C2>
            bool operator<=(const NDArray<T2, C2> &ndarray) const noexcept;
            
            template <typename T2, bool C2>
            bool operator>(const NDArray<T2, C2> &ndarray) const noexcept;
            
            template <typename T2, bool C2>
            bool operator<(const NDArray<T2, C2> &ndarray) const noexcept;

            // string function
            std::string str() const noexcept;

        private:
            // private utility functions
            template <typename T2, bool C2>
            const NDArray<T2, false> broadcast_expansion(const NDArray<T2, C2> &rhs) noexcept;

            const NDArray<T, false> axes_reorder(const Axes &axes) const noexcept;

        public:
            // getters
            inline const T* data() const noexcept {
                return this->m_data;
            }

            inline T& operator[](uint_fast64_t index) noexcept {
                return this->m_data[index];
            }

            inline const T& operator[](uint_fast64_t index) const noexcept {
                return this->m_data[index];
            }

            // arithmetical operators
            inline NDArray& operator+=(T value) noexcept {
                return this->add_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator+=(const NDArray<T2, C2> &rhs) {
                return this->add_eq(rhs);
            }

            inline NDArray& operator-=(T value) noexcept {
                return this->subtract_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator-=(const NDArray<T2, C2> &rhs) {
                return this->subtract_eq(rhs);
            }

            inline NDArray& operator*=(T value) noexcept {
                return this->multiply_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator*=(const NDArray<T2, C2> &rhs) {
                return this->multiply_eq(rhs);
            }

            inline NDArray& operator/=(T value) noexcept {
                return this->divide_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator/=(const NDArray<T2, C2> &rhs) {
                return this->divide_eq(rhs);
            }

            inline NDArray& operator%=(T value) noexcept {
                return this->remainder_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator%=(const NDArray<T2, C2> &rhs) {
                return this->remainder_eq(rhs);
            }
            
            inline NDArray& operator^=(T value) noexcept {
                return this->bit_xor_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator^=(const NDArray<T2, C2> &rhs) {
                return this->bit_xor_eq(rhs);
            }
            
            inline NDArray& operator&=(T value) noexcept {
                return this->bit_and_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator&=(const NDArray<T2, C2> &rhs) {
                return this->bit_and_eq(rhs);
            }
            
            inline NDArray& operator|=(T value) noexcept {
                return this->bit_or_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator|=(const NDArray<T2, C2> &rhs) {
                return this->bit_or_eq(rhs);
            }
            
            inline NDArray& operator<<=(T value) noexcept {
                return this->shl_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator<<=(const NDArray<T2, C2> &rhs) {
                return this->shl_eq(rhs);
            }
            
            inline NDArray& operator>>=(T value) noexcept {
                return this->shr_eq(value);
            }

            template <typename T2, bool C2>
            inline NDArray& operator>>=(const NDArray<T2, C2> &rhs) {
                return this->shr_eq(rhs);
            }
            
            inline NDArray& operator~() noexcept {
                return this->bit_not();
            }

            inline NDArray<T, true> operator+(T value) const noexcept {
                return this->add(value);
            }

            template <typename T2, bool C2>
            inline auto operator+(const NDArray<T2, C2> &rhs) const {
                return this->add(rhs);
            }

            inline NDArray<T, true> operator-(T value) const noexcept {
                return this->subtract(value);
            }

            template <typename T2, bool C2>
            inline auto operator-(const NDArray<T2, C2> &rhs) const {
                return this->subtract(rhs);
            }

            inline NDArray<T, true> operator*(T value) const noexcept {
                return this->multiply(value);
            }

            template <typename T2, bool C2>
            inline auto operator*(const NDArray<T2, C2> &rhs) const {
                return this->multiply(rhs);
            }

            inline NDArray<T, true> operator/(T value) const noexcept {
                return this->divide(value);
            }

            template <typename T2, bool C2>
            inline auto operator/(const NDArray<T2, C2> &rhs) const {
                return this->divide(rhs);
            }

            inline NDArray<T, true> operator%(T value) const noexcept {
                return this->remainder(value);
            }

            template <typename T2, bool C2>
            inline auto operator%(const NDArray<T2, C2> &rhs) const {
                return this->remainder(rhs);
            }

            inline NDArray<T, true> operator^(T value) const noexcept {
                return this->bit_xor(value);
            }

            template <typename T2, bool C2>
            inline auto operator^(const NDArray<T2, C2> &rhs) const {
                return this->bit_xor(rhs);
            }

            inline NDArray<T, true> operator&(T value) const noexcept {
                return this->bit_and(value);
            }

            template <typename T2, bool C2>
            inline auto operator&(const NDArray<T2, C2> &rhs) const {
                return this->bit_and(rhs);
            }

            inline NDArray<T, true> operator|(T value) const noexcept {
                return this->bit_or(value);
            }

            template <typename T2, bool C2>
            inline auto operator|(const NDArray<T2, C2> &rhs) const {
                return this->bit_or(rhs);
            }

            inline NDArray<T, true> operator<<(T value) const noexcept {
                return this->shl(value);
            }

            template <typename T2, bool C2>
            inline auto operator<<(const NDArray<T2, C2> &rhs) const {
                return this->shl(rhs);
            }

            inline NDArray<T, true> operator>>(T value) const noexcept {
                return this->shr(value);
            }

            template <typename T2, bool C2>
            inline auto operator>>(const NDArray<T2, C2> &rhs) const {
                return this->shr(rhs);
            }

            // more utility functions
            inline ArrayBase& base() {
                return *this;
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const NDArray &ndarray) noexcept {
                return stream << ndarray.str();
            }
    };
};

#include "src/ndlib/ndarray.tpp"
#endif