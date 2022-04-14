
#include "src/ndarray/nditerator.h"
#include "src/ndarray/ndarray_types.h"
#include <type_traits>

namespace laruen::ndarray {


    template <typename T>
    NDIterator<T>::NDIterator(T &ndarray) : ndarray(ndarray), index(0), ndindex(ndarray.ndim, 0) {
        static_assert(types::is_ndarray_v<T>, "NDIterator only support laruen::ndarray::NDArray");
    }

    template <typename T>
    auto& NDIterator<T>::next() {
        auto& value = this->ndarray[this->index];
        this->ndindex[this->ndarray.ndim - 1]++;
        this->index += this->ndarray.strides[ndarray.ndim - 1];
        
        for(uint8_t dim = this->ndarray.ndim;dim-- > 1;) {
            if(this->ndindex[dim] >= this->ndarray.shape[dim]) {
                this->ndindex[dim] = 0;
                this->ndindex[dim - 1]++;
                this->index += this->ndarray.strides[dim - 1] - this->ndarray.shape[dim] * this->ndarray.strides[dim];
            }
        }

        return value;
    }
};