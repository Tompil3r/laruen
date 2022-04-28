
#include "src/ndlib/ndlib.h"
#include "src/ndlib/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

using namespace laruen::math;

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

        NDArray<T, false> ndarray(lhs);
        uint8_t rhs_i = 0;
        uint8_t swap_i = lhs.m_ndim - rhs.m_ndim;

        for(uint8_t lhs_i = swap_i;lhs_i < lhs.m_ndim;lhs_i++) {
            if(lhs.m_shape[lhs_i] != rhs.m_shape[rhs_i]) {
                if(rhs.m_shape[rhs_i] == 1) {
                    std::swap(ndarray.m_shape[swap_i] ,ndarray.m_shape[lhs_i]);
                    std::swap(ndarray.m_strides[swap_i] ,ndarray.m_strides[lhs_i]);
                    swap_i++;
                }
                else {
                    throw std::invalid_argument("shapes cannot be broadcasted");
                }
            }
            
            rhs_i++;
        }

        return ndarray;
    }
}