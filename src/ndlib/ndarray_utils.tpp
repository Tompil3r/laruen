
#include "src/ndlib/ndarray_utils.h"
#include <cstdint>

namespace laruen::ndlib::utils {

    template <>
    uint8_t rev_count_diff<true>(const Shape &lhs, const Shape &rhs) noexcept {
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
    uint8_t rev_count_diff<false>(const Shape &lhs, const Shape &rhs) noexcept {
        return lhs.size() >= rhs.size() ? rev_count_diff<true>(lhs, rhs) : rev_count_diff<true>(rhs, lhs);
    }
}