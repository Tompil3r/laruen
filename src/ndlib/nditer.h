
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
    class NDIter {};

    template <typename T>
    class NDIter<T, true> {

        template <typename, bool> friend class NDArray;

        T &m_ndarray;
        uint_fast64_t m_index;

        public:
            NDIter(T &ndarray) noexcept
            : m_ndarray(ndarray), m_index(0)
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, uint_fast64_t index) noexcept
            : m_ndarray(ndarray), m_index(index)
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, const NDIndex &ndindex) noexcept
            : m_ndarray(ndarray), m_index(ndarray.ravel_ndindex(ndindex))
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            inline auto& next() noexcept {
                return this->m_ndarray.m_data[this->m_index++];
            }

            inline auto& next(uint_fast8_t axis) noexcept {
                auto &value = this->m_ndarray.m_data[this->m_index];
                this->m_index += this->m_ndarray.m_strides[axis];
                return value;
            }

            inline void reset() noexcept {
                this->m_index = 0;
            }

            inline void move_forward(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_index += amount * this->m_ndarray.m_strides[axis];
            }

            inline void move_forward(uint_fast64_t stride) noexcept {
                this->m_index += stride;
            }

            inline void forward_update(uint_fast8_t axis, uint_fast64_t amount) noexcept
            {}

            inline void move_backward(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_index -= amount * this->m_ndarray.m_strides[axis];
            }

            inline void move_backward(uint_fast64_t stride) noexcept {
                this->m_index -= stride;
            }

            inline void backward_update(uint_fast8_t axis, uint_fast64_t amount) noexcept
            {}
            
            inline void move(uint_fast8_t axis, int_fast64_t amount) noexcept {
                this->m_index += amount * this->m_ndarray.m_strides[axis];
            }

            inline void move(int_fast64_t stride) noexcept {
                this->m_index += stride;
            }

            inline void update(uint_fast8_t axis, int_fast64_t amount) noexcept
            {}

            inline void inc(uint_fast8_t axis) noexcept {
                this->m_index += this->m_ndarray.m_strides[axis];
            }

            inline void dec(uint_fast8_t axis) noexcept {
                this->m_index -= this->m_ndarray.m_strides[axis];
            }

            inline bool has_next() const noexcept {
                return this->m_index < this->m_ndarray.m_size;
            }

            inline auto& current() noexcept {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint_fast64_t& index() const noexcept {
                return this->m_index;
            }
    };

    template <typename T>
    class NDIter<T, false> {

        template <typename, bool> friend class NDArray;

        T &m_ndarray;
        uint_fast64_t m_index;
        NDIndex m_ndindex;
        const Strides &m_single_strides;

        public:
            NDIter(T &ndarray) noexcept
            : m_ndarray(ndarray), m_index(0), m_ndindex(ndarray.m_ndim, 0),
            m_single_strides(ndarray.forward_base()->strides())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, uint_fast64_t index) noexcept
            : m_ndarray(ndarray), m_index(index), m_ndindex(ndarray.unravel_index(index)),
            m_single_strides(ndarray.forward_base()->strides())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, const NDIndex &ndindex) noexcept
            : m_ndarray(ndarray), m_index(ndarray.ravel_ndindex(ndindex)), m_ndindex(ndindex),
            m_single_strides(ndarray.forward_base()->strides())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            NDIter(T &ndarray, NDIndex &&ndindex) noexcept
            : m_ndarray(ndarray), m_index(ndarray.ravel_ndindex(ndindex)), m_ndindex(std::move(ndindex)),
            m_single_strides(ndarray.forward_base()->strides())
            {
                static_assert(types::is_ndarray_v<T>, "NDIter only supports NDArray");
            }

            auto& next(uint_fast8_t axis) noexcept {
                auto &value = this->m_ndarray.m_data[this->m_index];
                this->m_ndindex[axis]++;
                this->m_index += this->m_ndarray.m_strides[axis];
                
                for(uint_fast8_t dim = axis;(dim > 0) && (this->m_ndindex[dim] >= this->m_ndarray.m_shape[dim]);) {
                    this->m_ndindex[dim] = 0;
                    dim--; // decrease dim "ahead of time" for minor efficiency improvements
                    this->m_ndindex[dim]++;
                    this->m_index += this->m_ndarray.m_strides[dim] - this->m_single_strides[dim];
                }

                return value;
            }

            inline auto& next() noexcept {
                return this->next(this->m_ndarray.m_ndim - 1);
            }

            void reset() noexcept {
                this->m_index = 0;
                uint_fast8_t ndim = this->m_ndindex.size();

                for(uint_fast8_t i = 0;i < ndim;i++) {
                    this->m_ndindex[i] = 0;
                }
            }

            inline void move_forward(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_index += amount * this->m_ndarray.m_strides[axis];
            }

            inline void move_forward(uint_fast64_t stride) noexcept {
                this->m_index += stride;
            }

            inline void forward_update(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_ndindex[axis] += amount;
            }

            inline void move_backward(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_index -= amount * this->m_ndarray.m_strides[axis];
            }

            inline void move_backward(uint_fast64_t stride) noexcept {
                this->m_index -= stride;
            }

            inline void backward_update(uint_fast8_t axis, uint_fast64_t amount) noexcept {
                this->m_ndindex[axis] -= amount;
            }
            
            inline void move(uint_fast8_t axis, int_fast64_t amount) noexcept {
                this->m_index += amount * this->m_ndarray.m_strides[axis];
            }

            inline void move(int_fast64_t stride) noexcept {
                this->m_index += stride;
            }

            inline void update(uint_fast8_t axis, int_fast64_t amount) noexcept {
                this->m_ndindex[axis] += amount;
            }

            inline void inc(uint_fast8_t axis) noexcept {
                this->m_index += this->m_ndarray.m_strides[axis];
            }

            inline void dec(uint_fast8_t axis) noexcept {
                this->m_index -= this->m_ndarray.m_strides[axis];
            }

            inline bool has_next() const noexcept {
                return this->m_ndindex[0] < this->m_ndarray.m_shape[0];
            }

            inline auto& current() noexcept {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint_fast64_t& index() const noexcept {
                return this->m_index;
            }

            inline const NDIndex& ndindex() const noexcept {
                return this->m_ndindex;
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