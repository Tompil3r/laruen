
#ifndef NDLIB_NDITER_H_
#define NDLIB_NDITER_H_

#include <type_traits>
#include <utility>
#include "src/multi/array_base.h"
#include "src/multi/types.h"
#include "src/multi/type_selection.h"

namespace laruen::multi {

    template <typename T>
    struct NDIter {
        T *ptr;
        const ArrayBase &arraybase;
        NDIndex ndindex;

        NDIter(T *data, const ArrayBase &arraybase) noexcept
        : ptr(data), arraybase(arraybase), ndindex(arraybase.ndim_, 0)
        {}

        T& next(uint_fast8_t axis) noexcept {
            T &value = *this->ptr;
            this->ndindex[axis]++;
            this->ptr += this->arraybase.strides_[axis];
            
            for(uint_fast8_t dim = axis;(dim > 0) && (this->ndindex[dim] >= this->arraybase.shape_[dim]);) {
                this->ndindex[dim] = 0;
                this->ptr -= this->arraybase.dim_sizes_[dim];
                dim--; // decrease dim "ahead of time" for minor efficiency improvements
                this->ndindex[dim]++;
                this->ptr += this->arraybase.strides_[dim];
            }

            return value;
        }

        inline T& next() noexcept {
            return this->next(this->arraybase.ndim_ - 1);
        }

        inline bool has_next() const noexcept {
            return this->ndindex[0] < this->arraybase.shape_[0];
        }

        inline T& current() noexcept {
            return *this->ptr;
        }
    };
};

#endif