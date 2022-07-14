
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
            
            template <bool CR>
            NDArray<uint_fast64_t>& indices_max(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept;

            NDArray<uint_fast64_t> indices_max(const Axes &axes) const noexcept;

            uint_fast64_t index_max() const noexcept;
            
            NDIndex ndindex_max() const noexcept;

            template <typename TR>
            NDArray<TR>& min(const Axes &axes, NDArray<TR> &out) const noexcept;
            
            NDArray<T> min(const Axes &axes) const noexcept;

            T min() const noexcept;
            
            template <bool CR>
            NDArray<uint_fast64_t>& indices_min(const Axes &axes, NDArray<uint_fast64_t> &out) const noexcept;

            NDArray<uint_fast64_t> indices_min(const Axes &axes) const noexcept;

            uint_fast64_t index_min() const noexcept;
            
            NDIndex ndindex_min() const noexcept;

            // indexing and slicing operators
            T& operator[](const NDIndex &ndindex) noexcept;

            const T& operator[](const NDIndex &ndindex) const noexcept;

            NDArray<T> operator[](const SliceRanges &ranges) noexcept;
            
            const NDArray<T> operator[](const SliceRanges &ranges) const noexcept;

            // arithmetical functions
            template <typename T2>
            NDArray& add_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& add_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& add_eq(const NDArray<T2> &rhs);

            NDArray& add_eq(T value) noexcept;
            
            template <typename TR>
            NDArray<TR>& add(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> add(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& add_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> add_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& add_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> add_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& add(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> add(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& subtract_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& subtract_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& subtract_eq(const NDArray<T2> &rhs);

            NDArray& subtract_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& subtract(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> subtract(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& subtract_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> subtract_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& subtract_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> subtract_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& subtract(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> subtract(const NDArray<T2> &rhs) const;
            
            template <typename T2>
            NDArray& multiply_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& multiply_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& multiply_eq(const NDArray<T2> &rhs);

            NDArray& multiply_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& multiply(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> multiply(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& multiply_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> multiply_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& multiply_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> multiply_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& multiply(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> multiply(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& divide_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& divide_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& divide_eq(const NDArray<T2> &rhs);

            NDArray& divide_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& divide(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> divide(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& divide_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> divide_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& divide_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> divide_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& divide(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> divide(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& bit_xor_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& bit_xor_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& bit_xor_eq(const NDArray<T2> &rhs);

            NDArray& bit_xor_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& bit_xor(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> bit_xor(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& bit_xor_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_xor_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_xor_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_xor_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_xor(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_xor(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& bit_and_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& bit_and_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& bit_and_eq(const NDArray<T2> &rhs);

            NDArray& bit_and_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& bit_and(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> bit_and(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& bit_and_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_and_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_and_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_and_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_and(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_and(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& bit_or_eq_b(const NDArray<T2> &rhs);
            
            template <typename T2>
            NDArray& bit_or_eq_r(const NDArray<T2> &rhs);
            
            template <typename T2>
            NDArray& bit_or_eq(const NDArray<T2> &rhs);

            NDArray& bit_or_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& bit_or(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> bit_or(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& bit_or_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_or_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_or_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_or_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& bit_or(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> bit_or(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& shl_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& shl_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& shl_eq(const NDArray<T2> &rhs);

            NDArray& shl_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& shl(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> shl(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& shl_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shl_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& shl_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shl_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& shl(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shl(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& shr_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& shr_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& shr_eq(const NDArray<T2> &rhs);

            NDArray& shr_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& shr(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> shr(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& shr_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shr_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& shr_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shr_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& shr(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> shr(const NDArray<T2> &rhs) const;

            NDArray& bit_not_eq() noexcept;

            template <typename TR>
            NDArray<TR>& bit_not(NDArray<TR> &out) const noexcept;

            NDArray<T> bit_not() const noexcept;

            template <typename T2>
            NDArray& remainder_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& remainder_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& remainder_eq(const NDArray<T2> &rhs);

            NDArray& remainder_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& remainder(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> remainder(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& remainder_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> remainder_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& remainder_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> remainder_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& remainder(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> remainder(const NDArray<T2> &rhs) const;

            template <typename T2>
            NDArray& power_eq_b(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& power_eq_r(const NDArray<T2> &rhs);

            template <typename T2>
            NDArray& power_eq(const NDArray<T2> &rhs);

            NDArray& power_eq(T value) noexcept;

            template <typename TR>
            NDArray<TR>& power(T value, NDArray<TR> &out) const noexcept;

            NDArray<T> power(T value) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& power_b(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> power_b(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& power_r(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> power_r(const NDArray<T2> &rhs) const;

            template <typename T2, typename TR>
            NDArray<TR>& power(const NDArray<T2> &rhs, NDArray<TR> &out) const;

            template <typename T2>
            NDArray<types::result_type_t<T, T2>> power(const NDArray<T2> &rhs) const;

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
            const NDArray<T2> broadcast_expansion(const NDArray<T2> &rhs) const noexcept;

            const NDArray<T> axes_reorder(const Axes &axes) const noexcept;

            template <typename T2, typename TR>
            NDArray<TR>& matmul_n3(const NDArray<T2> &rhs, NDArray<TR> &out) const noexcept;

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

            // arithmetical operators
            inline NDArray& operator+=(T value) noexcept {
                return this->add_eq(value);
            }

            template <typename T2>
            inline NDArray& operator+=(const NDArray<T2> &rhs) {
                return this->add_eq(rhs);
            }

            inline NDArray& operator-=(T value) noexcept {
                return this->subtract_eq(value);
            }

            template <typename T2>
            inline NDArray& operator-=(const NDArray<T2> &rhs) {
                return this->subtract_eq(rhs);
            }

            inline NDArray& operator*=(T value) noexcept {
                return this->multiply_eq(value);
            }

            template <typename T2>
            inline NDArray& operator*=(const NDArray<T2> &rhs) {
                return this->multiply_eq(rhs);
            }

            inline NDArray& operator/=(T value) noexcept {
                return this->divide_eq(value);
            }

            template <typename T2>
            inline NDArray& operator/=(const NDArray<T2> &rhs) {
                return this->divide_eq(rhs);
            }

            inline NDArray& operator%=(T value) noexcept {
                return this->remainder_eq(value);
            }

            template <typename T2>
            inline NDArray& operator%=(const NDArray<T2> &rhs) {
                return this->remainder_eq(rhs);
            }
            
            inline NDArray& operator^=(T value) noexcept {
                return this->bit_xor_eq(value);
            }

            template <typename T2>
            inline NDArray& operator^=(const NDArray<T2> &rhs) {
                return this->bit_xor_eq(rhs);
            }
            
            inline NDArray& operator&=(T value) noexcept {
                return this->bit_and_eq(value);
            }

            template <typename T2>
            inline NDArray& operator&=(const NDArray<T2> &rhs) {
                return this->bit_and_eq(rhs);
            }
            
            inline NDArray& operator|=(T value) noexcept {
                return this->bit_or_eq(value);
            }

            template <typename T2>
            inline NDArray& operator|=(const NDArray<T2> &rhs) {
                return this->bit_or_eq(rhs);
            }
            
            inline NDArray& operator<<=(T value) noexcept {
                return this->shl_eq(value);
            }

            template <typename T2>
            inline NDArray& operator<<=(const NDArray<T2> &rhs) {
                return this->shl_eq(rhs);
            }
            
            inline NDArray& operator>>=(T value) noexcept {
                return this->shr_eq(value);
            }

            template <typename T2>
            inline NDArray& operator>>=(const NDArray<T2> &rhs) {
                return this->shr_eq(rhs);
            }
            
            inline NDArray& operator~() noexcept {
                return this->bit_not();
            }

            inline NDArray<T> operator+(T value) const noexcept {
                return this->add(value);
            }

            template <typename T2>
            inline auto operator+(const NDArray<T2> &rhs) const {
                return this->add(rhs);
            }

            inline NDArray<T> operator-(T value) const noexcept {
                return this->subtract(value);
            }

            template <typename T2>
            inline auto operator-(const NDArray<T2> &rhs) const {
                return this->subtract(rhs);
            }

            inline NDArray<T> operator*(T value) const noexcept {
                return this->multiply(value);
            }

            template <typename T2>
            inline auto operator*(const NDArray<T2> &rhs) const {
                return this->multiply(rhs);
            }

            inline NDArray<T> operator/(T value) const noexcept {
                return this->divide(value);
            }

            template <typename T2>
            inline auto operator/(const NDArray<T2> &rhs) const {
                return this->divide(rhs);
            }

            inline NDArray<T> operator%(T value) const noexcept {
                return this->remainder(value);
            }

            template <typename T2>
            inline auto operator%(const NDArray<T2> &rhs) const {
                return this->remainder(rhs);
            }

            inline NDArray<T> operator^(T value) const noexcept {
                return this->bit_xor(value);
            }

            template <typename T2>
            inline auto operator^(const NDArray<T2> &rhs) const {
                return this->bit_xor(rhs);
            }

            inline NDArray<T> operator&(T value) const noexcept {
                return this->bit_and(value);
            }

            template <typename T2>
            inline auto operator&(const NDArray<T2> &rhs) const {
                return this->bit_and(rhs);
            }

            inline NDArray<T> operator|(T value) const noexcept {
                return this->bit_or(value);
            }

            template <typename T2>
            inline auto operator|(const NDArray<T2> &rhs) const {
                return this->bit_or(rhs);
            }

            inline NDArray<T> operator<<(T value) const noexcept {
                return this->shl(value);
            }

            template <typename T2>
            inline auto operator<<(const NDArray<T2> &rhs) const {
                return this->shl(rhs);
            }

            inline NDArray<T> operator>>(T value) const noexcept {
                return this->shr(value);
            }

            template <typename T2>
            inline auto operator>>(const NDArray<T2> &rhs) const {
                return this->shr(rhs);
            }

            // more utility functions
            inline ArrayBase& arraybase() {
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