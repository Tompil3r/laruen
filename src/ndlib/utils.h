
#ifndef NDLIB_UTILS_H_
#define NDLIB_UTILS_H_

#include <cstdint>
#include <stdexcept>
#include "src/ndlib/types.h"
#include "src/math/common.h"

namespace laruen::ndlib::utils {

    using laruen::ndlib::Shape, laruen::ndlib::Axes;

    template <bool = false>
    Shape broadcast(const Shape&, const Shape&);

    template <>
    Shape broadcast<true>(const Shape &lhs, const Shape &rhs) {
        // assume lhs.size() >= rhs.size()

        Shape bshape(lhs);
        const uint_fast8_t min_ndim = rhs.size();
        const uint_fast8_t ndim_delta = lhs.size() - min_ndim;
        uint_fast8_t imax, lval, rval;

        for(uint_fast8_t imin = 0;imin < min_ndim;imin++) {
            uint_fast8_t imax = imin + ndim_delta;
            lval = lhs[imax];
            rval = rhs[imin];

            if(lval != rval && lval != 1 && rval != 1) {
                throw std::invalid_argument("shapes cannot be broadcasted");
            }
            bshape[imax] = laruen::math::common::max(lval, rval);
        }

        return bshape;
    }

    template <>
    Shape broadcast<false>(const Shape &lhs, const Shape &rhs) {
        return lhs.size() >= rhs.size() ? broadcast<true>(lhs, rhs) : broadcast<true>(rhs, lhs);
    }


    Axes compress_axes(const Axes &axes, uint_fast8_t ndim) {
        Axes result(ndim - axes.size());
        uint_fast8_t ridx = 0;
        uint_fast8_t remaining_axes = -1;

        for(uint_fast8_t i = 0;i < axes.size();i++) {
            remaining_axes ^= 1 << axes[i];
        }

        for(uint_fast8_t i = 0;i < ndim;i++) {
            if(remaining_axes & (1 << i)) {
                result[ridx] = i;
                ridx++;
            }
        }

        return result;
    }

    inline uint_fast64_t ceil_index(float64_t index) noexcept {
        return (uint_fast64_t)index + ((uint_fast64_t)index < index);
    }
}


#endif