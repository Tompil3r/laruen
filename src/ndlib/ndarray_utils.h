
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndlib/ndarray_types.h"
#include <cstdint>


namespace laruen::ndlib::utils {
    template <bool = false> uint8_t rev_count_diff(const Shape&, const Shape&) noexcept;

    inline uint64_t ceil_index(float64_t index) noexcept {
        return (uint64_t)index + ((uint64_t)index < index);
    }
}




#include "src/ndlib/ndarray_utils.tpp"
#endif