#ifndef NDLIB_IMPL_H_
#define NDLIB_IMPL_H_

#include <cstdint>
#include "src/ndlib/nditer.h"
#include "src/ndlib/ndarray.h"
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

            uint_fast8_t rows_axis = lhs_base.ndim_ - 2;
            uint_fast8_t cols_axis = lhs_base.ndim_ - 1;

            if(lhs_base.ndim_ <= 2) {
                matmul_2d_n3(lhs_data, lhs_base.strides_[rows_axis], lhs_base.strides_[cols_axis],
                rhs_data, rhs_base.strides_[rows_axis], rhs_base.strides_[cols_axis], out_data, out_base.strides_[rows_axis],
                out_base.strides_[cols_axis], out_base.shape_[rows_axis], out_base.shape_[cols_axis], lhs_base.shape_[cols_axis]);

                return out_data;
            }

            uint_fast64_t stacks = (lhs_base.size_ / lhs_base.shape_[cols_axis]) / lhs_base.shape_[rows_axis];
            uint_fast8_t iteration_axis = lhs_base.ndim_ - 3;
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint_fast64_t stack = 0;stack < stacks;stack++) {
                matmul_2d_n3(lhs_iter.ptr, lhs_base.strides_[rows_axis], lhs_base.strides_[cols_axis],
                rhs_iter.ptr, rhs_base.strides_[rows_axis], rhs_base.strides_[cols_axis], out_iter.ptr, out_base.strides_[rows_axis],
                out_base.strides_[cols_axis], out_base.shape_[rows_axis], out_base.shape_[cols_axis], lhs_base.shape_[cols_axis]);

                lhs_iter.next(iteration_axis);
                rhs_iter.next(iteration_axis);
                out_iter.next(iteration_axis);
            }

            return out_data;
        }

        template <typename T, typename T2, typename TR>
        static TR* matmul(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data,
        const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base, uint_fast8_t depth) noexcept {
            /*
                function requirements:
                    - all arrays have already been broadcasted
                    - no 1d arrays
                    - all arrays are correct shape
            */

            if(!depth) {
                matmul_n3(lhs_data, lhs_base, rhs_data, rhs_base, out_data, out_base);
                return out_data;
            }

            uint_fast8_t rows_axis = lhs_base.ndim_ - 2;
            uint_fast8_t cols_axis = lhs_base.ndim_ - 1;

            uint_fast64_t total_rows = out_base.shape_[rows_axis]; // or lhs_base.shape_[rows_axis]
            uint_fast64_t total_cols = out_base.shape_[cols_axis]; // or rhs_base.shape_[cols_axis]
            uint_fast64_t total_shared = lhs_base.shape_[cols_axis]; // or rhs_base.shape_[rows_axis]

            ArrayBase lhs_q_base(lhs_base);
            ArrayBase rhs_q_base(rhs_base);
            ArrayBase out_q_base(out_base);

            lhs_q_base.shape_[rows_axis] >>= 1;
            lhs_q_base.shape_[cols_axis] >>= 1;
            lhs_q_base.dim_sizes_[rows_axis] >>= 1;
            lhs_q_base.dim_sizes_[cols_axis] >>= 1;
            lhs_q_base.size_ >>= 2;
            rhs_q_base.shape_[rows_axis] >>= 1;
            rhs_q_base.shape_[cols_axis] >>= 1;
            rhs_q_base.dim_sizes_[rows_axis] >>= 1;
            rhs_q_base.dim_sizes_[cols_axis] >>= 1;
            rhs_q_base.size_ >>= 2;
            out_q_base.shape_[rows_axis] >>= 1;
            out_q_base.shape_[cols_axis] >>= 1;
            out_q_base.dim_sizes_[rows_axis] >>= 1;
            out_q_base.dim_sizes_[cols_axis] >>= 1;
            out_q_base.size_ >>= 2;

            const T* const lhs_q11 = lhs_data;
            const T* const lhs_q12 = lhs_data + lhs_q_base.dim_sizes_[cols_axis];
            const T* const lhs_q21 = lhs_data + lhs_q_base.dim_sizes_[rows_axis];
            const T* const lhs_q22 = lhs_data + lhs_q_base.dim_sizes_[cols_axis] + lhs_q_base.dim_sizes_[rows_axis];

            const T2* const rhs_q11 = rhs_data;
            const T2* const rhs_q12 = rhs_data + rhs_q_base.dim_sizes_[cols_axis];
            const T2* const rhs_q21 = rhs_data + rhs_q_base.dim_sizes_[rows_axis];
            const T2* const rhs_q22 = rhs_data + rhs_q_base.dim_sizes_[cols_axis] + rhs_q_base.dim_sizes_[rows_axis];
            
            TR* const out_q11 = out_data;
            TR* const out_q12 = out_data + out_q_base.dim_sizes_[cols_axis];
            TR* const out_q21 = out_data + out_q_base.dim_sizes_[rows_axis];
            TR* const out_q22 = out_data + out_q_base.dim_sizes_[cols_axis] + out_q_base.dim_sizes_[rows_axis];
            
            // extra arrays for calculation memory
            NDArray<T> lhs_ext(lhs_q_base.shape_);
            NDArray<T2> rhs_ext(rhs_q_base.shape_);
            NDArray<TR> out_ext(out_q_base.shape_);

            depth--;

            matmul(lhs_q11, lhs_q_base, rhs_q11, rhs_q_base, out_q11, out_q_base, depth); // r1 -> out_q11
            subtract(lhs_q11, lhs_q_base, lhs_q21, lhs_q_base, lhs_ext.data_, lhs_ext); // s3 -> lhs_ext
            subtract(rhs_q22, rhs_q_base, rhs_q12, rhs_q_base, rhs_ext.data_, rhs_ext); // t3 -> rhs_ext
            matmul(lhs_ext.data_, lhs_ext, rhs_ext.data_, rhs_ext, out_ext.data_, out_ext, depth); // r7 -> out_ext
            add(lhs_q21, lhs_q_base, lhs_q22, lhs_q_base, lhs_ext.data_, lhs_ext); // s1 -> lhs_ext
            subtract(rhs_q12, rhs_q_base, rhs_q11, rhs_q_base, rhs_ext.data_, rhs_ext); // t1 -> rhs_ext
            matmul(lhs_ext.data_, lhs_ext, rhs_ext.data_, rhs_ext, out_q22, out_q_base, depth); // r5 -> out_q22
            subtract_eq(lhs_ext.data_, lhs_ext, lhs_q11, lhs_q_base); // s2 -> lhs_ext
            subtract(rhs_q22, rhs_q_base, rhs_ext.data_, rhs_ext, rhs_ext.data_, rhs_ext); // t2 -> lhs_ext
            matmul(lhs_ext.data_, lhs_ext, rhs_ext.data_, rhs_ext, out_q12, out_q_base, depth); // r6 -> out_q12
            subtract(lhs_q12, lhs_q_base, lhs_ext.data_, lhs_ext, lhs_ext.data_, lhs_ext); // s4 -> lhs_ext
            subtract_eq(rhs_ext.data_, rhs_ext, rhs_q21, rhs_q_base); // t4 -> rhs_ext
            add_eq(out_q12, out_q_base, out_q11, out_q_base); // c2 -> out_q12
            add_eq(out_ext.data_, out_ext, out_q12, out_q_base); // c3 -> out_ext
            add_eq(out_q12, out_q_base, out_q22, out_q_base); // c4 -> out_q12
            add_eq(out_q22, out_q_base, out_ext.data_, out_ext); // c7 -> out_q22 (DONE, 1/4)
            matmul(lhs_ext.data_, lhs_ext, rhs_q22, rhs_q_base, out_q21, out_q_base, depth); // r3 -> out_q21
            add_eq(out_q12, out_q_base, out_q21, out_q_base); // c5 -> out_q12 (DONE, 2/4)
            matmul(lhs_q12, lhs_q_base, rhs_q21, rhs_q_base, out_q21, out_q_base, depth); // r2 -> out_q21
            add_eq(out_q11, out_q_base, out_q21, out_q_base); // c1 -> out_q11 (DONE, 3/4)
            matmul(lhs_q22, lhs_q_base, rhs_ext.data_, rhs_ext, out_q21, out_q_base, depth); // r4 -> out_q21
            subtract(out_ext.data_, out_ext, out_q21, out_q_base, out_q21, out_q_base); // c6 -> out_q21 (DONE, 4/4)

            return out_data;
        }

        template <typename T, typename T2>
        static T* add_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() += rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* add_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() += value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* add(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() + rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* subtract_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() -= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* subtract_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() -= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* subtract(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() - rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* multiply_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() *= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* multiply_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() *= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* multiply(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() * rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* divide_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */

            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() /= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* divide_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() /= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* divide(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() / rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* bit_xor_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() ^= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* bit_xor_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() ^= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_xor(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() ^ rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* bit_and_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() &= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* bit_and_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() &= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_and(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() & rhs_iter.next();
            }

            return out_data;
        }
        
        template <typename T, typename T2>
        static T* bit_or_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() |= rhs_iter.next();
            }

            return lhs_data;
        }
        
        template <typename T>
        static T* bit_or_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() |= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_or(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() | rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* shl_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() <<= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* shl_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() <<= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* shl(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() << rhs_iter.next();
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* shr_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() >>= rhs_iter.next();
            }

            return lhs_data;
        }

        template <typename T>
        static T* shr_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() >>= value;
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* shr(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = lhs_iter.next() >> rhs_iter.next();
            }

            return out_data;
        }

        template <typename T>
        static T* bit_not_eq(T *data, const ArrayBase &base) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() = ~iter.current();
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* bit_not(const T *lhs_data, const ArrayBase &lhs_base, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint_fast64_t i = 0 ;i < lhs_base.size_;i++) {
                out_iter.next() = ~lhs_iter.next();
            }
            
            return out_data;
        }

        template <typename T, typename T2>
        static T* remainder_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            /* implementation function: arrays must be broadcasted if needed */
            
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() = laruen::math::common::remainder(lhs_iter.current(), rhs_iter.next());
            }

            return lhs_data;
        }

        template <typename T>
        static T* remainder_eq(T *data, const ArrayBase &base, T value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() = laruen::math::common::remainder(iter.current(), value);
            }

            return data;
        }

        template <typename T, typename TR>
        static TR* remainder(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), rhs_iter.next());
            }

            return out_data;
        }

        template <typename T, typename T2>
        static T* power_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter rhs_iter(rhs_data, rhs_base);

            for(uint_fast64_t i = 0;i < lhs_base.size_;i++) {
                lhs_iter.next() = laruen::math::common::pow(lhs_iter.current(), rhs_iter.next());
            }

            return lhs_data;
        }

        template <typename T, typename T2>
        static T* power_eq(T *data, const ArrayBase &base, T2 value) noexcept {
            NDIter iter(data, base);

            for(uint_fast64_t i = 0;i < base.size_;i++) {
                iter.next() = laruen::math::common::pow(iter.current(), value);
            }

            return data;
        }

        template <typename T, typename T2, typename TR>
        static TR* power(const T *lhs_data, const ArrayBase &lhs_base, T2 value, TR *out_data, const ArrayBase &out_base) noexcept {
            NDIter lhs_iter(lhs_data, lhs_base);
            NDIter out_iter(out_data, out_base);

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
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

            for(uint64_t i = 0;i < lhs_base.size_;i++) {
                out_iter.next() = laruen::math::common::pow(lhs_iter.next(), rhs_iter.next());
            }

            return out_data;
        }
    };
}

#endif