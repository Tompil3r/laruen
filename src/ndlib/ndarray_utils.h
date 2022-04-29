
#ifndef NDARRAY_UTILS_H
#define NDARRAY_UTILS_H

#include "src/ndlib/ndarray_types.h"
#include <cstdint>


namespace laruen::ndlib::utils {
    inline uint64_t ceil_index(float64_t index) {
        return (uint64_t)index + ((uint64_t)index < index);
    }

    template <bool = false> uint8_t rev_count_diff(const Shape&, const Shape&);

    template <>
    uint8_t rev_count_diff<true>(const Shape &lhs, const Shape &rhs) {
        // assume lhs.size() >= rhs.size()

        uint8_t count = 0;
        uint8_t lidx = lhs.size() - rhs.size();

        for(uint8_t ridx = 0;ridx < rhs.size();ridx++) {
            count += lhs[lidx] != rhs[ridx];
            lidx++;
        }
        
        return count;
    }

    template <>
    inline uint8_t rev_count_diff(const Shape &lhs, const Shape &rhs) {
        return lhs.size() >= rhs.size() ? rev_count_diff<true>(lhs, rhs) : rev_count_diff<true>(rhs, lhs);
    }
}





#endif