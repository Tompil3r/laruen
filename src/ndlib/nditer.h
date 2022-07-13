
#ifndef NDLIB_NDITER_H_
#define NDLIB_NDITER_H_

#include <type_traits>
#include <utility>
#include "src/ndlib/array_base.h"
#include "src/ndlib/types.h"
#include "src/ndlib/type_selection.h"

namespace laruen::ndlib {

    template <typename T>
    struct NDIter {
        T *ptr;
        const ArrayBase &arraybase;
        NDIndex ndindex;

        NDIter(T *data, const ArrayBase &arraybase) noexcept
        : ptr(data), arraybase(arraybase), ndindex(arraybase.m_ndim, 0)
        {}

        T& next(uint_fast8_t axis) noexcept {
            T &value = *this->ptr;
            this->ndindex[axis]++;
            this->ptr += this->arraybase.m_strides[axis];
            
            for(uint_fast8_t dim = axis;(dim > 0) && (this->ndindex[dim] >= this->arraybase.m_shape[dim]);) {
                this->ndindex[dim] = 0;
                this->ptr -= this->arraybase.m_dim_sizes[dim];
                dim--; // decrease dim "ahead of time" for minor efficiency improvements
                this->ndindex[dim]++;
                this->ptr += this->arraybase.m_strides[dim];
            }

            return value;
        }

        inline T& next() noexcept {
            return this->next(this->arraybase.m_ndim - 1);
        }

        inline bool has_next() const noexcept {
            return this->ndindex[0] < this->arraybase.m_shape[0];
        }

        inline T& current() noexcept {
            return *this->ptr;
        }
    };
};

#endif