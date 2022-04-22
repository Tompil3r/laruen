
#include "src/ndarray/nditerator.h"
#include "src/ndarray/ndarray_types.h"
#include <type_traits>

namespace laruen::ndarray {

    template <template <typename, bool> typename A, typename T>
    NDIterator<A, T, true>::NDIterator(A<T, true> &ndarray) : m_ndarray(ndarray), m_index(0) {
        static_assert(types::is_ndarray_v<A<T, true>>, "NDIterator only support laruen::ndarray::NDArray");
    }

    template <template <typename, bool> typename A, typename T>
    NDIterator<A, T, false>::NDIterator(A<T, false> &ndarray)
    : m_ndarray(ndarray), m_index(0), m_ndindex(ndarray.m_ndim, 0)
    {
        static_assert(types::is_ndarray_v<A<T, false>>, "NDIterator only support laruen::ndarray::NDArray");
    }

    template <template <typename, bool> typename A, typename T>
    auto& NDIterator<A, T, false>::next() {
        bool check_next = true;
        auto& value = this->m_ndarray[this->m_index];
        this->m_ndindex[this->m_ndarray.m_ndim - 1]++;
        this->m_index += this->m_ndarray.m_strides[this->m_ndarray.m_ndim - 1];
        
        for(uint8_t dim = this->m_ndarray.m_ndim;check_next && dim-- > 1;) {
            if(check_next = this->m_ndindex[dim] >= this->m_ndarray.m_shape[dim]) {
                this->m_ndindex[dim] = 0;
                this->m_ndindex[dim - 1]++;
                this->m_index += this->m_ndarray.m_strides[dim - 1] - this->m_ndarray.m_shape[dim] * this->m_ndarray.m_strides[dim];
            }
        }

        return value;
    }

    template <template <typename, bool> typename A, typename T>
    void NDIterator<A, T, false>::reset() {
        this->m_index = 0;
        uint8_t ndim = this->m_ndindex.size();

        for(uint8_t i = 0;i < ndim;i++) {
            this->m_ndindex[i] = 0;
        }
    }
};