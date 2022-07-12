
#ifndef NDLIB_NDITER_H_
#define NDLIB_NDITER_H_

#include <type_traits>
#include <utility>
#include "src/ndlib/types.h"
#include "src/ndlib/ndarray.h"
#include "src/ndlib/type_selection.h"

namespace laruen::ndlib {
    template <typename T, bool C> class NDArray;

    template <typename T, bool C = T::CONTIGUOUS>
    struct NDIter {};

    template <typename T>
    struct NDIter<T, true> {
        template <typename, bool> friend class NDArray;
        typedef std::conditional_t<std::is_const_v<T>, const typename T::DType, typename T::DType> DType;

        T &ndarray;
        DType *ptr;

        public:
            NDIter(T &ndarray) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data)
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, uint_fast64_t index) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data + index)
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, const NDIndex &ndindex) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data + ndarray.ravel_ndindex(ndindex))
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            inline DType& next() noexcept {
                return *(this->ptr++);
            }

            inline DType& next(uint_fast8_t axis) noexcept {
                DType &value = *this->ptr;
                this->ptr += this->ndarray.m_strides[axis];
                return value;
            }

            inline void reset() noexcept {
                this->ptr = this->ndarray.m_data;
            }

            inline bool has_next() const noexcept {
                return this->ptr < (this->ndarray.m_data + this->ndarray.m_size);
            }

            inline auto& current() noexcept {
                return *this->ptr;
            }
    };

    template <typename T>
    struct NDIter<T, false> {
        template <typename, bool> friend class NDArray;
        typedef std::conditional_t<std::is_const_v<T>, const typename T::DType, typename T::DType> DType;

        T &ndarray;
        DType *ptr;
        NDIndex m_ndindex;
        const Strides m_dim_sizes;

        public:
            NDIter(T &ndarray) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data), m_ndindex(ndarray.m_ndim, 0),
            m_dim_sizes(ndarray.dim_sizes())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, uint_fast64_t index) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data + index), m_ndindex(ndarray.unravel_index(index)),
            m_dim_sizes(ndarray.dim_sizes())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, const NDIndex &ndindex) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data + ndarray.ravel_ndindex(ndindex)), m_ndindex(ndindex),
            m_dim_sizes(ndarray.dim_sizes())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, NDIndex &&ndindex) noexcept
            : ndarray(ndarray), ptr(ndarray.m_data + ndarray.ravel_ndindex(ndindex)), m_ndindex(std::move(ndindex)),
            m_dim_sizes(ndarray.dim_sizes())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            DType& next(uint_fast8_t axis) noexcept {
                DType &value = *this->ptr;
                this->m_ndindex[axis]++;
                this->ptr += this->ndarray.m_strides[axis];
                
                for(uint_fast8_t dim = axis;(dim > 0) && (this->m_ndindex[dim] >= this->ndarray.m_shape[dim]);) {
                    this->m_ndindex[dim] = 0;
                    this->ptr -= this->m_dim_sizes[dim];
                    dim--; // decrease dim "ahead of time" for minor efficiency improvements
                    this->m_ndindex[dim]++;
                    this->ptr += this->ndarray.m_strides[dim];
                }

                return value;
            }

            inline DType& next() noexcept {
                return this->next(this->ndarray.m_ndim - 1);
            }

            void reset() noexcept {
                this->ptr = this->ndarray.m_data;
                uint_fast8_t ndim = this->m_ndindex.size();

                for(uint_fast8_t i = 0;i < ndim;i++) {
                    this->m_ndindex[i] = 0;
                }
            }

            inline bool has_next() const noexcept {
                return this->m_ndindex[0] < this->ndarray.m_shape[0];
            }

            inline DType* current() noexcept {
                return *this->ptr;
            }
    };

    template <typename T>
    inline auto nditer_begin(T &ndarray) noexcept {
        return NDIter(ndarray);
    }

    template <typename T> NDIter(T&) -> NDIter<T, T::CONTIGUOUS>;
    template <typename T> NDIter(T&, uint_fast64_t) -> NDIter<T, T::CONTIGUOUS>;
    template <typename T> NDIter(T&, const NDIndex&) -> NDIter<T, T::CONTIGUOUS>;
    template <typename T> NDIter(T&, NDIndex&&) -> NDIter<T, T::CONTIGUOUS>;
};

#endif