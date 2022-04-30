
#include "src/ndlib/ndlib.h"
#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/ndarray_utils.h"
#include "src/math/common.h"
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

using namespace laruen::math;
using namespace laruen;

namespace laruen::ndlib {

    Shape dir_broadcast(const Shape &shape1, const Shape &shape2) {
        // assume shape1.size() >= shape2.size()

        Shape bshape(shape1);
        const uint8_t min_ndim = shape2.size();
        const uint8_t ndim_delta = shape1.size() - min_ndim;
        uint8_t idx_max, value1, value2;

        for(uint8_t idx_min = 0;idx_min < min_ndim;idx_min++) {
            uint8_t idx_max = idx_min + ndim_delta;
            value1 = shape1[idx_max];
            value2 = shape2[idx_min];

            if(value1 != value2 && value1 != 1 && value2 != 1) {
                throw std::invalid_argument("shapes cannot be broadcasted");
            }
            bshape[idx_max] = common::max(value1, value2);
        }

        return bshape;
    }

    Shape broadcast(const Shape &shape1, const Shape &shape2) {
        return shape1.size() >= shape2.size() ? dir_broadcast(shape1, shape2) : dir_broadcast(shape2, shape1);
    }

    template <typename T, bool C, typename T2, bool C2>
    NDArray<T, false> broadcast_reorder(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs) {
        if(rhs.m_ndim > lhs.m_ndim) {
            throw std::invalid_argument("shapes cannot be broadcasted");
        }

        NDArray<T, false> reorder(lhs.m_ndim, lhs.m_data, false, lhs.m_size);
        uint8_t lidx = lhs.m_ndim - rhs.m_ndim;
        uint8_t low_priority_idx = lidx;
        uint8_t high_priority_idx = lidx + ndlib::utils::rev_count_diff<true>(lhs.m_shape, rhs.m_shape);

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