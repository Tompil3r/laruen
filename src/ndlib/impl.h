#ifndef NDLIB_STATIC_H_
#define NDLIB_STATIC_H_

#include <cstdint>

namespace laruen::ndlib::impl {

    template <typename TR, typename T, typename T2>
    inline TR dot_1d(T *lhs_ptr, uint_fast64_t lhs_stride,
    T2 *rhs_ptr, uint_fast64_t rhs_stride, uint_fast64_t size) noexcept {
        TR product = 0;
        for(uint_fast64_t i = 0;i < size;i++) {
            product += (*lhs_ptr) * (*rhs_ptr);
            lhs_ptr += lhs_stride;
            rhs_ptr += rhs_stride;
        }

        return product;
    }

    template <typename T, typename T2, typename TR>
    TR* matmul_2d_n3(T *lhs_ptr, uint_fast64_t lhs_row_stride, uint_fast64_t lhs_col_stride,
    T2 *rhs_ptr, uint_fast64_t rhs_row_stride, uint_fast64_t rhs_col_stride,
    TR *out_ptr, uint_fast64_t out_row_stride, uint_fast64_t out_col_stride,
    uint_fast64_t rows, uint_fast64_t cols, uint_fast64_t shared)
    {
        T2 *rhs_start_ptr = rhs_ptr;
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
}

#endif