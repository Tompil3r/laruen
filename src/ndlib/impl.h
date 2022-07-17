#ifndef NDLIB_STATIC_H_
#define NDLIB_STATIC_H_

#include <cstdint>
#include "src/ndlib/nditer.h"
#include "src/math/common.h"

namespace laruen::ndlib {

    struct Impl {
        
        template <typename TR, typename T, typename T2>
        static inline TR dot_1d(const T *lhs_ptr, uint_fast64_t lhs_stride,
        const T2 *rhs_ptr, uint_fast64_t rhs_stride, uint_fast64_t size) noexcept {
            TR product = 0;
            for(uint_fast64_t i = 0;i < size;i++) {
                product += (*lhs_ptr) * (*rhs_ptr);
                lhs_ptr += lhs_stride;
                rhs_ptr += rhs_stride;
            }

            return product;
        }

        template <typename T, typename T2, typename TR>
        static TR* matmul_2d_n3(const T *lhs_ptr, uint_fast64_t lhs_row_stride, uint_fast64_t lhs_col_stride,
        const T2 *rhs_ptr, uint_fast64_t rhs_row_stride, uint_fast64_t rhs_col_stride,
        TR *out_ptr, uint_fast64_t out_row_stride, uint_fast64_t out_col_stride,
        uint_fast64_t rows, uint_fast64_t cols, uint_fast64_t shared)
        {
            const T2 *rhs_start_ptr = rhs_ptr;
            TR *out_start_ptr = out_ptr;
            TR *out_checkpoint = out_ptr;

            for(uint_fast64_t row = 0;row < rows;row++) {
                for(uint_fast64_t col = 0;col < cols;col++) {
                    *out_ptr = dot_1d<TR>(lhs_ptr, lhs_col_stride, rhs_ptr, rhs_row_stride, shared);
                    out_ptr += out_col_stride;
                    rhs_ptr += rhs_col_stride;
                }
                out_checkpoint += out_row_stride;
                out_ptr = out_checkpoint;
                lhs_ptr += lhs_row_stride;
                rhs_ptr = rhs_start_ptr;
            }

            return out_start_ptr;
        }

        template <typename T, typename T2, typename TR>
        static TR* matmul_n3(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data,
        const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) noexcept {
            /* assumes the following:
                - shapes are valid
                - no broadcasting needed
                - min ndim == 2 (no dot product)
            */
            using laruen::ndlib::NDIter;

            uint_fast8_t rows_axis = lhs_base.m_ndim - 2;
            uint_fast8_t cols_axis = lhs_base.m_ndim - 1;

            if(lhs_base.m_ndim <= 2) {
                matmul_2d_n3(lhs_data, lhs_base.m_strides[rows_axis], lhs_base.m_strides[cols_axis],
                rhs_data, rhs_base.m_strides[rows_axis], rhs_base.m_strides[cols_axis], out_data, out_base.m_strides[rows_axis],
                out_base.m_strides[cols_axis], out_base.m_shape[rows_axis], out_base.m_shape[cols_axis], lhs_base.m_shape[cols_axis]);

                return out_data;
            }

            uint_fast64_t stacks = (lhs_base.m_size / lhs_base.m_shape[cols_axis]) / lhs_base.m_shape[rows_axis];
            uint_fast8_t iteration_axis = lhs_base.m_ndim - 3;
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint_fast64_t stack = 0;stack < stacks;stack++) {
                matmul_2d_n3(lhs_iter.ptr, lhs_base.m_strides[rows_axis], lhs_base.m_strides[cols_axis],
                rhs_iter.ptr, rhs_base.m_strides[rows_axis], rhs_base.m_strides[cols_axis], out_iter.ptr, out_base.m_strides[rows_axis],
                out_base.m_strides[cols_axis], out_base.m_shape[rows_axis], out_base.m_shape[cols_axis], lhs_base.m_shape[cols_axis]);

                lhs_iter.next(iteration_axis);
                rhs_iter.next(iteration_axis);
                out_iter.next(iteration_axis);
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* add_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() += rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* add_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() += value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* add(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() + value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* add(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() + rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* subtract_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() -= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* subtract_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() -= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* subtract(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() - value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* subtract(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() - rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* multiply_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() *= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* multiply_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() *= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* multiply(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() * value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* multiply(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() * rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* divide_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() /= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* divide_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() /= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* divide(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() / value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* divide(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() / rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* bit_xor_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() ^= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* bit_xor_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() ^= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_xor(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() ^ value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* bit_xor(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() ^ rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* bit_and_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() &= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* bit_and_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() &= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_and(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() & value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* bit_and(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() & rhs_iter.next();
            }

            return out_data;
        }
        
        template <typename T, typename T2>
        static T* bit_or_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() |= rhs_iter.next();
            }

            return lhs_data;
        }
        
        template <typename T>
        static T* bit_or_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() |= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_or(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() | value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* bit_or(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() | rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* shl_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() <<= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* shl_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() <<= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* shl(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() << value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* shl(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() << rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* shr_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() >>= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* shr_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() >>= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* shr(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() >> value;
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* shr(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = lhs_iter.next() >> rhs_iter.next();
            }

            return out_data;
        }

        template <typename T>
        static T* bit_not_eq(T *data, const ArrayBase &base) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() = ~iter.current();
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_not(const T *lhs_data, const ArrayBase &lhs_base, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint_fast64_t i = 0 ;i < lhs_base.m_size;i++) {
                out_iter.next() = ~lhs_iter.next();
            }
            
            return out_data;
        }

        template <typename T, typename T2>
        static T* remainder_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() = laruen::math::common::remainder(lhs_iter.current(), rhs_iter.next());
            }

            return lhs_data;
        }

        template <typename T>
        static T* remainder_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() = laruen::math::common::remainder(iter.current(), value);
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* remainder(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), value);
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* remainder(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), rhs_iter.next());
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* power_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.m_size;i++) {
                lhs_iter.next() = laruen::math::common::pow(lhs_iter.current(), rhs_iter.next());
            }

            return lhs_data;
        }

        template <typename T>
        static T* power_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.m_size;i++) {
                iter.next() = laruen::math::common::pow(iter.current(), value);
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* power(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = laruen::math::common::pow(lhs_iter.next(), value);
            }
            
            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* power(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.m_size;i++) {
                out_iter.next() = laruen::math::common::pow(lhs_iter.next(), rhs_iter.next());
            }

            return out_data;
        }
    };
}

#endif