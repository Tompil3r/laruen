
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndlib/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>

namespace laruen::ndlib { template <typename T, bool C> class NDArray; }
using laruen::ndlib::NDArray;

namespace laruen::ndlib::utils {

    template <bool = false>
    Shape broadcast(const Shape&, const Shape&);

    Axes remaining_axes(const Axes &axes, uint_fast8_t ndim);

    inline uint_fast64_t ceil_index(float64_t index) noexcept {
        return (uint_fast64_t)index + ((uint_fast64_t)index < index);
    }
}


#include "src/ndlib/ndarray_utils.tpp"
#endif