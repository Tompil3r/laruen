
#include "src/ndlib/ndarray_utils.h"
#include "src/ndlib/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>

using namespace laruen;

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

    template <>
    Shape broadcast<true>(const Shape &lhs, const Shape &rhs) {
        // assume lhs.size() >= rhs.size()

        Shape bshape(lhs);
        const uint8_t min_ndim = rhs.size();
        const uint8_t ndim_delta = lhs.size() - min_ndim;
        uint8_t imax, lval, rval;

        for(uint8_t imin = 0;imin < min_ndim;imin++) {
            uint8_t imax = imin + ndim_delta;
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

    template <typename T, bool C, typename T2, bool C2>
    NDArray<T, false> broadcast_reorder(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs) {
        if(rhs.m_ndim > lhs.m_ndim) {
            throw std::invalid_argument("shapes cannot be broadcasted");
        }

        NDArray<T, false> reorder(lhs.m_data, Shape(lhs.m_ndim), Strides(lhs.m_ndim),
        lhs.m_size, lhs.m_ndim, false);
        uint8_t lidx = lhs.m_ndim - rhs.m_ndim;
        uint8_t low_priority_idx = lidx;
        uint8_t high_priority_idx = lidx + rev_count_diff<true>(lhs.m_shape, rhs.m_shape);

        for(uint8_t i = 0;i < lidx;i++) {
            reorder.m_shape[i] = lhs.m_shape[i];
            reorder.m_strides[i] = lhs.m_strides[i];
        }

        for(uint8_t ridx = 0;ridx < rhs.m_ndim;ridx++) {
            if(lhs.m_shape[lidx] != rhs.m_shape[ridx]) {
                if(rhs.m_shape[ridx] == 1) {
                    reorder.m_shape[low_priority_idx] = lhs.m_shape[lidx];
                    reorder.m_strides[low_priority_idx] = lhs.m_strides[lidx];
                    low_priority_idx++;
                }
                else {
                    throw std::invalid_argument("shapes cannot be broadcasted");
                }
            }
            else {
                reorder.m_shape[high_priority_idx] = lhs.m_shape[lidx];
                reorder.m_strides[high_priority_idx] = lhs.m_strides[lidx];
                high_priority_idx++;
            }
            lidx++;
        }

        return reorder;
    }
}