
#ifndef LARUEN_MULTI_NDITER_H_
#define LARUEN_MULTI_NDITER_H_

#include <type_traits>
#include <utility>
#include "src/multi/ndarray.h"
#include "src/multi/array_base.h"
#include "src/multi/types.h"
#include "src/multi/type_selection.h"

namespace laruen::multi {

    template <typename> class NDArray;

    template <typename T>
    struct NDIter {
        T *ptr;
        const ArrayBase &arraybase;
        NDIndex ndindex;

        NDIter(T *data, const ArrayBase &arraybase) noexcept
        : ptr(data), arraybase(arraybase), ndindex(arraybase.ndim_, 0)
        {}

        NDIter(NDArray<T> &ndarray) noexcept
        : ptr(ndarray.data_), arraybase(ndarray), ndindex(ndarray.ndim_, 0)
        {}

        template <typename TT>
        NDIter(const NDArray<TT> &ndarray) noexcept
        : ptr(std::as_const(ndarray.data_)), arraybase(ndarray), ndindex(ndarray.ndim_, 0)
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
            return this->ndindex.front() < this->arraybase.shape_.front();
        }

        inline T& current() noexcept {
            return *this->ptr;
        }

        inline T& operator*() noexcept {
            return this->current();
        }

        inline T& operator++(int) noexcept {
            return this->next();
        }

        inline T& operator++() noexcept {
            this->next();
            return this->current();
        }
    };

    template <typename T> NDIter(NDArray<T>&) -> NDIter<T>;
    template <typename TT> NDIter(const NDArray<TT>&) -> NDIter<const TT>;
};

#endif