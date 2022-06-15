
#ifndef NDLIB_UTILS_H_
#define NDLIB_UTILS_H_

#include "src/ndlib/types.h"
#include "src/math/common.h"
#include <cstdint>

namespace laruen::ndlib::utils {

    template <bool = false>
    Shape broadcast(const Shape&, const Shape&);

    Axes compress_axes(const Axes &axes, uint_fast8_t ndim);

    inline uint_fast64_t ceil_index(float64_t index) noexcept {
        return (uint_fast64_t)index + ((uint_fast64_t)index < index);
    }
}


#include "src/ndlib/utils.tpp"
#endif