
#include "src/ndarray/ndarray_lib.h"
#include "src/ndarray/ndarray_types.h"
#include "src/math/common.h"
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

using namespace laruen::math;

namespace laruen::ndarray {

    Shape d_broadcast(const Shape &shape1, const Shape &shape2) {
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
        return shape1.size() >= shape2.size() ? d_broadcast(shape1, shape2) : d_broadcast(shape2, shape1);
    }

    bool equal_dims(const Shape &shape1, const Shape &shape2) {
        uint8_t ndim;
        bool eq_dims = (ndim = shape1.size()) == shape2.size();

        for(uint8_t dim = 0;dim < ndim && eq_dims;dim++) {
            eq_dims = (shape1[dim] == shape2[dim]);
        }

        return eq_dims;
    }
}