
#ifndef NDARRAY_H
#define NDARRAY_H

#include "src/ndlib/ndarray_utils.h"
#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/nditerator.h"
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

    template <typename T = float64_t, bool C = true> class NDArray : public ArrayBase {
        template <typename, bool> friend class NDArray;
        friend class NDIterator<T, C>;
        friend class ConstNDIterator<T, C>;

        private:
            T *m_data;

        public:
            typedef ArrayBase Base;
            typedef T DType;
            static constexpr bool CONTIGUOUS = C;

            ~NDArray();
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
            NDArray(NDArray &ndarray, const Axes &axes) noexcept;
            template <bool C2> NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2> || C != C2>> NDArray(const NDArray<T2, C2> &ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2> || C != C2>> NDArray(NDArray<T2, C2> &&ndarray) noexcept;

            NDArray& operator=(const NDArray &ndarray) noexcept;
            NDArray& operator=(NDArray &&ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(const NDArray<T2, C2> &ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(NDArray<T2, C2> &&ndarray) noexcept;

            template <typename T2, bool C2> void copy_data_from(const NDArray<T2, C2> &ndarray) noexcept;
            void fill(T value) noexcept;

            T max() const noexcept;
            uint_fast64_t index_max() const noexcept;
            NDIndex ndindex_max() const noexcept;
            T min() const noexcept;
            uint_fast64_t index_min() const noexcept;
            NDIndex ndindex_min() const noexcept;

            T& operator[](const NDIndex &ndindex) noexcept;
            const T& operator[](const NDIndex &ndindex) const noexcept;
            NDArray<T, false> operator[](const SliceRanges &ranges) noexcept;
            const NDArray<T, false> operator[](const SliceRanges &ranges) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator+=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator-=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator*=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator/=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator+(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator-(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator*(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator/(T2 value) const noexcept;
            template <typename T2, bool C2> auto operator+(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator-(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator*(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator/(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> NDArray& operator+=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator-=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator*=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator/=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> bool operator==(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator!=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator>=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator<=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator>(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator<(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2> NDArray& operator^=(T2 value) noexcept;
            template <typename T2> NDArray& operator&=(T2 value) noexcept;
            template <typename T2> NDArray& operator|=(T2 value) noexcept;
            template <typename T2> NDArray& operator<<=(T2 value) noexcept;
            template <typename T2> NDArray& operator>>=(T2 value) noexcept;
            template <typename T2> auto operator^(T2 value) const noexcept;
            template <typename T2> auto operator&(T2 value) const noexcept;
            template <typename T2> auto operator|(T2 value) const noexcept;
            template <typename T2> auto operator<<(T2 value) const noexcept;
            template <typename T2> auto operator>>(T2 value) const noexcept;
            NDArray operator~() const noexcept;
            template <typename T2, bool C2> NDArray& operator^=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator&=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator|=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator<<=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator>>=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> auto operator^(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator&(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator|(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator<<(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator>>(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value) noexcept;
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value) noexcept;
            template <typename T2> auto operator%(T2 value) const noexcept;
            template <typename T2, bool C2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> auto operator%(const NDArray<T2, C2> &ndarray) const;

            std::string str() const noexcept;

        private:
            template <typename T2, bool C2>
            const NDArray<T2, false> broadcast_expansion(const NDArray<T2, C2> &rhs) noexcept;
            
            template <auto Op, typename T2>
            NDArray& invoke_value_assignment(T2 value) noexcept;

            template <auto Op, typename T2, bool C2>
            NDArray& invoke_normal_assignment(const NDArray<T2, C2> &rhs) noexcept;

            template <auto Op, typename T2, bool C2>
            inline NDArray& invoke_broadcast_assignment(const NDArray<T2, C2> &rhs) {
                return this->invoke_normal_assignment<Op>(this->broadcast_expansion(rhs));
            }

            template <bool B, auto Op, typename T2, bool C2>
            inline NDArray& invoke_ndarray_assignment(const NDArray<T2, C2> &rhs) {
                if constexpr(B) {
                    return this->invoke_broadcast_assignment<Op>(rhs);
                }
                else {
                    return this->invoke_normal_assignment<Op>(rhs);
                }
            }

            template <auto Op, typename T2, typename TR>
            NDArray<TR, true> invoke_new_value(T2 value) const noexcept;

        public:
            inline const T* data() const noexcept {
                return this->m_data;
            }

            inline T& operator[](uint_fast64_t index) noexcept {
                return this->m_data[index];
            }

            inline const T& operator[](uint_fast64_t index) const noexcept {
                return this->m_data[index];
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const NDArray &ndarray) noexcept {
                return stream << ndarray.str();
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& add_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::addition<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& add_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->add_eq<false>(rhs) : this->add_eq<true>(rhs);
            }

            inline NDArray& add_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::addition<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& subtract_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::subtraction<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& subtract_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->subtract_eq<false>(rhs) : this->subtract_eq<true>(rhs);
            }

            inline NDArray& subtract_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::subtraction<T, T>>(value);
            }
            
            template <bool B, typename T2, bool C2>
            inline NDArray& multiply_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::multiplication<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& multiply_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->multiply_eq<false>(rhs) : this->multiply_eq<true>(rhs);
            }

            inline NDArray& multiply_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::multiplication<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& divide_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::division<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& divide_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->divide_eq<false>(rhs) : this->divide_eq<true>(rhs);
            }

            inline NDArray& divide_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::division<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& bit_xor_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::bit_xor<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& bit_xor_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->bit_xor_eq<false>(rhs) : this->bit_xor_eq<true>(rhs);
            }

            inline NDArray& bit_xor_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::bit_xor<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& bit_and_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::bit_and<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& bit_and_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->bit_and_eq<false>(rhs) : this->bit_and_eq<true>(rhs);
            }

            inline NDArray& bit_and_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::bit_and<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& bit_or_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::bit_or<T, T2>>(rhs);
            }
            
            template <typename T2, bool C2>
            inline NDArray& bit_or_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->bit_or_eq<false>(rhs) : this->bit_or_eq<true>(rhs);
            }

            inline NDArray& bit_or_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::bit_or<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& shl_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::bit_shl<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& shl_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->shl_eq<false>(rhs) : this->shl_eq<true>(rhs);
            }

            inline NDArray& shl_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::bit_shl<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& shr_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::bit_shr<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& shr_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->shr_eq<false>(rhs) : this->shr_eq<true>(rhs);
            }

            inline NDArray& shr_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::bit_shr<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& remainder_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::remainder<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& remainder_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->remainder_eq<false>(rhs) : this->remainder_eq<true>(rhs);
            }

            inline NDArray& remainder_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::remainder<T, T>>(value);
            }

            template <bool B, typename T2, bool C2>
            inline NDArray& power_eq(const NDArray<T2, C2> &rhs) {
                return this->invoke_ndarray_assignment<B, ndlib::utils::operations::power<T, T2>>(rhs);
            }

            template <typename T2, bool C2>
            inline NDArray& power_eq(const NDArray<T2, C2> &rhs) {
                return this->m_shape == rhs.m_shape ? this->power_eq<false>(rhs) : this->power_eq<true>(rhs);
            }

            inline NDArray& power_eq(T value) noexcept {
                return this->invoke_value_assignment<ndlib::utils::operations::power<T, T>>(value);
            }

            /* ----- ndlib -----  */
            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, false> ndlib::utils::broadcast_reorder(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <auto Op, typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& invoke_broadcast_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <auto Op, typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& invoke_normal_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) noexcept;

            template <auto Op, typename T1, bool C1, typename T2>
            friend NDArray<T1, C1>& invoke_value_assignment(NDArray<T1, C1> &lhs, T2 value) noexcept;
    };
    
    template <typename T, bool C> NDArray(NDArray<T, C>&, const Axes&) -> NDArray<T, false>;
};

#include "src/ndlib/ndarray.tpp"
#endif