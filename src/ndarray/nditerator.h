
#ifndef NDITERATOR_H
#define NDITERATOR_H

#include "src/ndarray/ndarray_types.h"
#include "src/ndarray/ndarray.h"
#include <type_traits>

namespace laruen::ndarray {
    template <typename T, bool C> class NDArray;

    /*
        code duplication is necessary to allow template deduction
        otherwise constructor syntax becomes quite verbose
    */
    template <typename T, bool C> class NDIterator {};
    template <typename T, bool C> class ConstNDIterator {};

    template <typename T>
    class NDIterator<T, true> {
        NDArray<T, true> &m_ndarray;
        uint64_t m_index;

        public:
            NDIterator() = delete;

            NDIterator(NDArray<T, true> &ndarray) : m_ndarray(ndarray), m_index(0) {}

            inline T& next() {
                return this->m_ndarray.m_data[this->m_index++];
            }

            inline void reset() {
                this->m_index = 0;
            }

            inline bool has_next() const {
                return this->m_index < this->m_ndarray.m_size;
            }

            inline T& current() {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint64_t& index() const {
                return this->m_index;
            }
    };

    template <typename T>
    class NDIterator<T, false> {
        NDArray<T, false> &m_ndarray;
        uint64_t m_index;
        NDIndex m_ndindex;

        public:
            NDIterator() = delete;

            NDIterator(NDArray<T, false> &ndarray)
            : m_ndarray(ndarray), m_index(0), m_ndindex(ndarray.m_ndim, 0) {}

            T& next() {
                auto& value = this->m_ndarray.m_data[this->m_index];
                this->m_ndindex[this->m_ndarray.m_ndim - 1]++;
                this->m_index += this->m_ndarray.m_strides[this->m_ndarray.m_ndim - 1];
                
                for(uint8_t dim = this->m_ndarray.m_ndim;(dim-- > 1) && (this->m_ndindex[dim] >= this->m_ndarray.m_shape[dim]);) {
                    this->m_ndindex[dim] = 0;
                    this->m_ndindex[dim - 1]++;
                    this->m_index += this->m_ndarray.m_strides[dim - 1] - this->m_ndarray.m_shape[dim] * this->m_ndarray.m_strides[dim];
                }

                return value;
            }

            void reset() {
                this->m_index = 0;
                uint8_t ndim = this->m_ndindex.size();

                for(uint8_t i = 0;i < ndim;i++) {
                    this->m_ndindex[i] = 0;
                }
            }

            inline bool has_next() const {
                return this->m_ndindex[0] < this->m_ndarray.m_shape[0];
            }

            inline T& current() {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint64_t& index() const {
                return this->m_index;
            }

            inline const NDIndex& ndindex() const {
                return this->m_ndindex;
            }
    };

    template <typename T>
    class ConstNDIterator<T, true> {
        const NDArray<T, true> &m_ndarray;
        uint64_t m_index;

        public:
            ConstNDIterator() = delete;

            ConstNDIterator(const NDArray<T, true> &ndarray) : m_ndarray(ndarray), m_index(0) {}

            inline const T& next() {
                return this->m_ndarray.m_data[this->m_index++];
            }

            inline void reset() {
                this->m_index = 0;
            }

            inline bool has_next() const {
                return this->m_index < this->m_ndarray.m_size;
            }

            inline const T& current() {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint64_t& index() const {
                return this->m_index;
            }
    };

    template <typename T>
    class ConstNDIterator<T, false> {
        const NDArray<T, false> &m_ndarray;
        uint64_t m_index;
        NDIndex m_ndindex;

        public:
            ConstNDIterator() = delete;

            ConstNDIterator(const NDArray<T, false> &ndarray)
            : m_ndarray(ndarray), m_index(0), m_ndindex(ndarray.m_ndim, 0) {}

            const T& next() {
                auto& value = this->m_ndarray.m_data[this->m_index];
                this->m_ndindex[this->m_ndarray.m_ndim - 1]++;
                this->m_index += this->m_ndarray.m_strides[this->m_ndarray.m_ndim - 1];
                
                for(uint8_t dim = this->m_ndarray.m_ndim;(dim-- > 1) && (this->m_ndindex[dim] >= this->m_ndarray.m_shape[dim]);) {
                    this->m_ndindex[dim] = 0;
                    this->m_ndindex[dim - 1]++;
                    this->m_index += this->m_ndarray.m_strides[dim - 1] - this->m_ndarray.m_shape[dim] * this->m_ndarray.m_strides[dim];
                }

                return value;
            }

            void reset() {
                this->m_index = 0;
                uint8_t ndim = this->m_ndindex.size();

                for(uint8_t i = 0;i < ndim;i++) {
                    this->m_ndindex[i] = 0;
                }
            }

            inline bool has_next() const {
                return this->m_ndindex[0] < this->m_ndarray.m_shape[0];
            }

            inline const T& current() {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint64_t& index() const {
                return this->m_index;
            }

            inline const NDIndex& ndindex() const {
                return this->m_ndindex;
            }
    };

    template <typename T> NDIterator(NDArray<T, true>&) -> NDIterator<T, true>;
    template <typename T> NDIterator(NDArray<T, false>&) -> NDIterator<T, false>;
    template <typename T> ConstNDIterator(const NDArray<T, true>&) -> ConstNDIterator<T, true>;
    template <typename T> ConstNDIterator(const NDArray<T, false>&) -> ConstNDIterator<T, false>;
};

#endif