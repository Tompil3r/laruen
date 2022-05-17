
#ifndef BASE_ARRAY_H
#define BASE_ARRAY_H

#include "src/ndlib/ndlib_types.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>
#include <ostream>

namespace laruen::ndlib {

    class ArrayBase {
        protected:
            Shape m_shape;
            Strides m_strides;
            uint_fast64_t m_size;
            uint_fast8_t m_ndim;
            bool m_free_mem;


        public:
            ArrayBase() noexcept = default;

            ArrayBase(const Shape &shape, const Strides &strides, uint_fast64_t size,
            uint_fast8_t ndim, bool free_mem) noexcept
            : m_shape(shape), m_strides(strides), m_size(size), m_ndim(ndim), m_free_mem(free_mem) {}

            ArrayBase(Shape &&shape, Strides &&strides, uint_fast64_t size,
            uint_fast8_t ndim, bool free_mem) noexcept
            : m_shape(std::move(shape)), m_strides(std::move(strides)), m_size(size), m_ndim(ndim), m_free_mem(free_mem) {}

            ArrayBase(uint_fast8_t ndim, bool free_mem = true, uint_fast64_t size = 0) noexcept
            : m_shape(ndim), m_strides(ndim), m_size(size), m_ndim(ndim), m_free_mem(free_mem) {}

            ArrayBase(const Shape &shape, bool free_mem = true) noexcept
            : m_shape(shape), m_strides(shape.size()), m_ndim(shape.size()), m_free_mem(free_mem)
            {
                uint_fast64_t stride = 1;
                this->m_size = (this->m_ndim > 0);
                
                for(uint_fast8_t dim = this->m_ndim; dim-- > 0;) {
                    this->m_strides[dim] = stride;
                    this->m_size *= shape[dim];
                    stride *= shape[dim];
                }
            }

            ArrayBase(const ArrayBase &base, bool free_mem) noexcept
            : m_shape(base.m_shape), m_strides(base.m_strides), m_size(base.m_size),
            m_ndim(base.m_ndim), m_free_mem(free_mem) {}

            void reshape(const Shape &shape) {
                uint_fast64_t prev_size = this->m_size;
                this->m_ndim = shape.size();
                this->m_shape = shape;
                this->m_strides.resize(this->m_ndim);
                this->m_size = (this->m_ndim > 0);

                uint_fast64_t stride = 1;
                
                for(uint_fast8_t dim = this->m_ndim; dim-- > 0;) {
                    this->m_strides[dim] = stride;
                    this->m_size *= shape[dim];
                    stride *= shape[dim];
                }

                if(this->m_size != prev_size) {
                    throw std::invalid_argument("invalid shape - number of elements do not match");
                }
            }

            uint_fast64_t ravel_ndindex(const NDIndex &ndindex) const noexcept {
                uint_fast64_t index = 0;
                uint_fast8_t ndim = ndindex.size();

                for(uint_fast8_t dim = 0;dim < ndim;dim++) {
                    index += ndindex[dim] * this->m_strides[dim];
                }

                return index;
            }

            NDIndex unravel_index(uint_fast64_t index) const noexcept {
                NDIndex ndindex(this->m_ndim);

                for(uint_fast8_t dim = 0;dim < this->m_ndim;dim++) {
                    ndindex[dim] = index / this->m_strides[dim];
                    index -= ndindex[dim] * this->m_strides[dim];
                }

                return ndindex;
            }

            void squeeze() noexcept {
                uint_fast8_t new_ndim = 0;

                for(uint_fast8_t dim = 0;dim < this->m_ndim;dim++) {
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

            std::string str() const noexcept {
                std::string str("shape = (");
                uint_fast8_t dim = 0;

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

            uint_fast64_t physical_index(uint_fast64_t logical_index) const noexcept {
                uint_fast64_t cstride = this->m_size;
                uint_fast64_t physical_index = 0;
                uint_fast64_t dim_index;

                for(uint_fast8_t dim = 0;dim < this->m_ndim;dim++) {
                    cstride /= this->m_shape[dim];
                    dim_index = logical_index / cstride;
                    physical_index += this->m_strides[dim] * dim_index;
                    logical_index -= cstride * dim_index;
                }

                return physical_index;
            }

            inline const Shape& shape() const noexcept {
                return this->m_shape;
            }

            inline const Strides& strides() const noexcept {
                return this->m_strides;
            }

            inline const uint_fast64_t& size() const noexcept {
                return this->m_size;
            }

            inline const uint_fast8_t& ndim() const noexcept {
                return this->m_ndim;
            }

            inline const bool& free_mem() const noexcept {
                return this->m_free_mem;
            }

            inline void modify_mem_release(bool free_mem) noexcept {
                this->m_free_mem = free_mem;
            }

            friend inline std::ostream& operator<<(std::ostream &stream, const ArrayBase &base) noexcept {
                return stream << base.str();
            }
    };
}

#endif