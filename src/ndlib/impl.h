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
}

#endif