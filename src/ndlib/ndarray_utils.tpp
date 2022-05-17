
#include "src/ndlib/ndarray_utils.h"
#include "src/ndlib/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>

using namespace laruen;

namespace laruen::ndlib::utils {

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
            bshape[imax] = math::common::max(lval, rval);
        }

        return bshape;
    }

    template <>
    Shape broadcast<false>(const Shape &lhs, const Shape &rhs) {
        return lhs.size() >= rhs.size() ? broadcast<true>(lhs, rhs) : broadcast<true>(rhs, lhs);
    }
}