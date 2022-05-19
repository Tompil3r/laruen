
#ifndef NDITER_H
#define NDITER_H

#include "src/ndlib/ndlib_types.h"
#include "src/ndlib/ndarray.h"
#include <type_traits>

namespace laruen::ndlib {
    template <typename T, bool C> class NDArray;

    template <typename T, bool C = T::CONTIGUOUS>
    class NDIter {
        
    };

    template <typename T>
    class NDIter<T, true> {
        T &m_ndarray;
        uint_fast64_t m_index;

        public:
            NDIter(T &ndarray) noexcept
            : m_ndarray(ndarray), m_index(0)
            {}

            inline auto& next() noexcept {
                return this->m_ndarray.m_data[this->m_index++];
            }

            inline void reset() noexcept {
                this->m_index = 0;
            }

            inline bool has_next() const noexcept {
                return this->m_index < this->m_ndarray.m_size;
            }

            inline T& current() noexcept {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint_fast64_t& index() const noexcept {
                return this->m_index;
            }
    };

    template <typename T>
    class NDIter<T, false> {
        T &m_ndarray;
        uint_fast64_t m_index;
        NDIndex m_ndindex;

        public:
            NDIter(T &ndarray) noexcept
            : m_ndarray(ndarray), m_index(0), m_ndindex(ndarray.m_ndim, 0)
            {}

            auto& next() noexcept {
                auto &value = this->m_ndarray.m_data[this->m_index];
                this->m_ndindex[this->m_ndarray.m_ndim - 1]++;
                this->m_index += this->m_ndarray.m_strides[this->m_ndarray.m_ndim - 1];
                
                for(uint_fast8_t dim = this->m_ndarray.m_ndim;(dim-- > 1) && (this->m_ndindex[dim] >= this->m_ndarray.m_shape[dim]);) {
                    this->m_ndindex[dim] = 0;
                    this->m_ndindex[dim - 1]++;
                    this->m_index += this->m_ndarray.m_strides[dim - 1] - this->m_ndarray.m_shape[dim] * this->m_ndarray.m_strides[dim];
                }

                return value;
            }

            void reset() noexcept {
                this->m_index = 0;
                uint_fast8_t ndim = this->m_ndindex.size();

                for(uint_fast8_t i = 0;i < ndim;i++) {
                    this->m_ndindex[i] = 0;
                }
            }

            inline bool has_next() const noexcept {
                return this->m_ndindex[0] < this->m_ndarray.m_shape[0];
            }

            inline T& current() noexcept {
                return this->m_ndarray.m_data[this->m_index];
            }

            inline const uint_fast64_t& index() const noexcept {
                return this->m_index;
            }

            inline const NDIndex& ndindex() const noexcept {
                return this->m_ndindex;
            }
    };

    template <typename T> NDIter(T&) -> NDIter<T, T::CONTIGUOUS>;
};

#endif