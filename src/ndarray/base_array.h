
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>
#include <utility>

class BaseArray {
    protected:
        Shape m_shape;
        Strides m_strides;
        uint64_t m_size;
        uint8_t m_ndim;
        bool m_free_mem;


    public:
        BaseArray(const Shape &shape) : m_shape(shape),
        m_strides(shape.size()), m_ndim(shape.size()), m_free_mem(true)
        {
            uint64_t stride = 1;
            this->m_size = (this->m_ndim > 0);
            
            for(uint8_t dim = this->m_ndim; dim-- > 0;) {
                this->m_strides[dim] = stride;
                this->m_size *= shape[dim];
                stride *= shape[dim];
            }
        }
};






#endif