
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndarray/ndarray_types.h"
#include <cstdint>
#include <utility>
#include <stdexcept>

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

        void reshape(const Shape &shape) {
            uint64_t prev_size = this->m_size;
            this->m_ndim = shape.size();
            this->m_shape = shape;
            this->m_strides.resize(this->m_ndim);
            this->m_size = (this->m_ndim > 0);

            uint64_t stride = 1;
            
            for(uint8_t dim = this->m_ndim; dim-- > 0;) {
                this->m_strides[dim] = stride;
                this->m_size *= shape[dim];
                stride *= shape[dim];
            }

            if(this->m_size != prev_size) {
                throw std::invalid_argument("invalid shape - number of elements do not match");
            }
        }

        uint64_t ravel_ndindex(const NDIndex &ndindex) const {
            uint64_t index = 0;
            uint8_t ndim = ndindex.size();

            for(uint8_t dim = 0;dim < ndim;dim++) {
                index += ndindex[dim] * this->m_strides[dim];
            }

            return index;
        }

        NDIndex unravel_index(uint64_t index) const {
            NDIndex ndindex(this->m_ndim);

            for(uint8_t dim = 0;dim < this->m_ndim;dim++) {
                ndindex[dim] = index / this->m_strides[dim];
                index -= ndindex[dim] * this->m_strides[dim];
            }

            return ndindex;
        }

        inline const Shape& shape() const {
            return this->m_shape;
        }

        inline const Strides& strides() const {
            return this->m_strides;
        }

        inline const uint64_t& size() const {
            return this->m_size;
        }

        inline const uint8_t& ndim() const {
            return this->m_ndim;
        }

        inline const bool& free_mem() const {
            return this->m_free_mem;
        }

        inline void modify_mem_release(bool free_mem) {
            this->m_free_mem = free_mem;
        }
};






#endif