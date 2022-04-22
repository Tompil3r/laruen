
#ifndef NDITERATOR_H
#define NDITERATOR_H

namespace laruen::ndarray {

    template <template <typename, bool> typename A, typename T, bool C>
    class NDIterator {
        public:
            NDIterator() = delete;
            NDIterator(A<T, C>&);
    };

    template <template <typename, bool> typename A, typename T>
    class NDIterator<A, T, true> {
        A<T, true> &m_ndarray;
        uint64_t m_index;

        public:
            NDIterator() = delete;
            NDIterator(A<T, true> &ndarray);

            inline auto& next() {
                return this->m_ndarray.m_data[this->m_index++];
            }

            inline void reset() {
                this->m_index = 0;
            }

            inline bool has_next() const {
                return this->m_index < this->m_ndarray.m_size;
            }
    };

    template <template <typename, bool> typename A, typename T>
    class NDIterator<A, T, false> {
        A<T, false> &m_ndarray;
        uint64_t m_index;
        NDIndex m_ndindex;

        public:
            NDIterator() = delete;
            NDIterator(A<T, false> &ndarray);

            auto& next();
            void reset();

            inline bool has_next() const {
                return this->m_ndindex[0] < this->m_ndarray.m_shape[0];
            }
    };

};

#include "src/ndarray/nditerator.tpp"
#endif