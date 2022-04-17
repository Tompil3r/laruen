
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>
#include <utility>

class ArrayBase {
    protected:
        Shape m_shape;
        Strides m_strides;
        uint64_t m_size;
        uint8_t m_ndim;
        bool m_free_mem;


    public:
        ArrayBase(const Shape &shape) : m_shape(shape),
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

        inline const Shape& shape() {
            return this->m_shape;
        }

        inline const Strides& strides() {
            return this->m_strides;
        }

        inline const uint64_t& size() {
            return this->m_size;
        }

        inline const uint8_t& ndim() {
            return this->m_ndim;
        }

        inline const bool& free_mem() {
            return this->m_free_mem;
        }
};






#endif