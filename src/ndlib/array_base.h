
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndlib/ndarray_types.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>

class ArrayBase {
    protected:
        Shape m_shape;
        Strides m_strides;
        uint64_t m_size;
        uint8_t m_ndim;
        bool m_free_mem;


    public:
        ArrayBase() = default;

        ArrayBase(uint8_t ndim, bool free_mem = true) : m_shape(ndim),
        m_strides(ndim), m_ndim(ndim), m_free_mem(free_mem) {}

        ArrayBase(const Shape &shape, bool free_mem = true) : m_shape(shape),
        m_strides(shape.size()), m_ndim(shape.size()), m_free_mem(free_mem)
        {
            uint64_t stride = 1;
            this->m_size = (this->m_ndim > 0);
            
            for(uint8_t dim = this->m_ndim; dim-- > 0;) {
                this->m_strides[dim] = stride;
                this->m_size *= shape[dim];
                stride *= shape[dim];
            }
        }

        ArrayBase(const ArrayBase &base, bool free_mem) : m_shape(base.m_shape),
        m_strides(base.m_strides), m_size(base.m_size), m_ndim(base.m_ndim), m_free_mem(free_mem) {}

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

        void squeeze() {
            uint8_t new_ndim = 0;

            for(uint8_t dim = 0;dim < this->m_ndim;dim++) {
                if(this->m_shape[dim] > 1) {
                    this->m_shape[new_ndim] = this->m_shape[dim];
                    this->m_strides[new_ndim] = this->m_strides[dim];
                    new_ndim++;
                }
            }

            this->m_ndim = new_ndim;
            this->m_shape.resize(this->m_ndim);
            this->m_strides.resize(this->m_ndim);
        }

        bool equal_dims(const ArrayBase &base) const {
            bool equal = this->m_ndim == base.m_ndim;

            for(uint8_t dim = 0;equal && dim < this->m_ndim;dim++) {
                equal = this->m_shape[dim] == base.m_shape[dim];
            }

            return equal;
        }

        std::string str() const {
            std::string str("shape = (");
            uint8_t dim = 0;

            for(dim = 0;dim < this->m_ndim - 1;dim++) {
                str += std::to_string(this->m_shape[dim]);
                str.push_back(',');
                str.push_back(' ');
            }
            str += std::to_string(this->m_shape[dim]) + ")\nstrides = (";

            for(dim = 0;dim < this->m_ndim - 1;dim++) {
                str += std::to_string(this->m_strides[dim]);
                str.push_back(',');
                str.push_back(' ');
            }
            str += std::to_string(this->m_strides[dim]) + ")\nsize = " + 
            std::to_string(this->m_size) + "\nndim = " +
            std::to_string(this->m_ndim);
            str.push_back('\n');

            return str;
        }

        uint64_t physical_index(uint64_t logical_index) const {
            uint64_t cstride = this->m_size;
            uint64_t physical_index = 0;
            uint64_t dim_index;

            for(uint8_t dim = 0;dim < this->m_ndim;dim++) {
                cstride /= this->m_shape[dim];
                dim_index = logical_index / cstride;
                physical_index += this->m_strides[dim] * dim_index;
                logical_index -= cstride * dim_index;
            }

            return physical_index;
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