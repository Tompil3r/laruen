
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