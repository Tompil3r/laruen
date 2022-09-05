
#ifndef NDLIB_UTILS_H_
#define NDLIB_UTILS_H_

#include <cstdint>
#include <stdexcept>
#include "src/multi/types.h"
#include "src/math/common.h"

namespace laruen::multi::utils {

    namespace impl {

        using laruen::multi::Shape;
        using laruen::multi::Axes;

        Shape broadcast_(const Shape &lhs, const Shape &rhs) {
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

        Shape matmul_broadcast_(const Shape &lhs, const Shape &rhs) {
            // assume lhs.size() >= rhs.size()

            Shape bshape(lhs);
            const uint_fast8_t min_ndim = rhs.size();
            const uint_fast8_t ndim_delta = lhs.size() - min_ndim;
            uint_fast8_t imax, lval, rval;

            for(uint_fast8_t imin = 0;imin < min_ndim - 2;imin++) {
                uint_fast8_t imax = imin + ndim_delta;
                lval = lhs[imax];
                rval = rhs[imin];

                if(lval != rval && lval != 1 && rval != 1) {
                    throw std::invalid_argument("shapes cannot be broadcasted");
                }
                bshape[imax] = laruen::math::common::max(lval, rval);
            }

            bshape.back() = rhs.back();
            return bshape;
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

        Shape retrieve_axes(const Shape &shape, const Axes &axes) {
            Shape result(axes.size());

            for(uint_fast8_t i = 0;i < axes.size();i++) {
                result[i] = shape[axes[i]];
            }

            return result;
        }

        inline Shape broadcast(const Shape &lhs, const Shape &rhs) {
            return lhs.size() >= rhs.size() ? broadcast_(lhs, rhs) : broadcast_(rhs, lhs);
        }

        inline Shape matmul_broadcast(const Shape &lhs, const Shape &rhs) {
            return lhs.size() >= rhs.size() ? matmul_broadcast_(lhs, rhs) : matmul_broadcast_(rhs, lhs);
        }

        inline uint_fast64_t ceil_index(float64_t index) noexcept {
            return (uint_fast64_t)index + ((uint_fast64_t)index < index);
        }
    }

    using namespace impl;
}


#endif