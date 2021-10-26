
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/typenames.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>


template <typename T> class NDArray
{
    T *data;
    Shape shape;
    Strides strides;
    uint8_t ndim;
    uint64_t size;

        
    public:
        NDArray();
        NDArray(const Shape &shape);
        NDArray(const Shape &shape, T fill_value);
        NDArray(T start, T stop, T step);
        
        const T* get_data();


        ~NDArray()
        {
            delete this->data;
        }


        const Shape& get_shape() const
        {
            return this->shape;
        }


        const Strides& get_strides() const
        {
            return this->strides;
        }


        uint8_t get_ndim() const
        {
            return this->ndim;
        }


        uint64_t get_size() const
        {
            return this->size;
        }


        void construct_shape(const Shape &shape)
        {
            this->ndim = shape.size();
            uint64_t stride = 1;
            uint64_t size = shape[this->ndim - 1];

            this->strides = Strides(this->ndim);
            this->strides[this->ndim - 1] = stride;
            
            for(uint8_t dim = this->ndim - 1;dim-- > 0;)
            {
                stride *= shape[dim + 1];
                this->strides[dim] = stride;
                size *= shape[dim];
            }

            this->shape = Shape(shape);
            this->size = size;
        }


        void reshape(const Shape &shape)
        {
            uint64_t stride = this->strides[this->ndim - 1];
            this->ndim = shape.size();
            uint64_t size = shape[this->ndim - 1];

            this->strides = Strides(this->ndim);
            this->strides[this->ndim - 1] = stride;

            for(uint8_t dim = this->ndim - 1;dim-- > 0;)
            {
                stride *= shape[dim + 1];
                this->strides[dim] = stride;
                size *= shape[dim];
            }

            assert(this->size == size);
            this->shape = Shape(shape);
        }


        void print(uint8_t dim = 0, uint64_t data_index = 0, bool not_first = false, bool not_last = true) const
        {
            uint32_t dim_idx;
            uint64_t stride;

            if(not_first) std::cout << std::string(dim, ' '); 
            std::cout << '[';

            if(dim == this->ndim - 1)
            {
                stride = this->strides[dim];

                for(dim_idx = 0;dim_idx < this->shape[dim] - 1;dim_idx++)
                {
                    std::cout << this->data[data_index] << ',' << ' ';
                    data_index += stride;
                }

                std::cout << this->data[data_index] << ']';
                if(not_last) std::cout << '\n';
                
                return;
            }

            this->print(dim + 1, data_index, false, true);
            data_index += this->strides[dim];            

            for(dim_idx = 1;dim_idx < this->shape[dim] - 1;dim_idx++)
            {
                this->print(dim + 1, data_index, true, true);
                data_index += this->strides[dim];
            }

            this->print(dim + 1, data_index, true, false);

            std::cout << ']';
            
            if(!dim) std::cout << '\n';
            else if(not_last) std::cout << std::string(this->ndim - dim, '\n');
        }
};

#endif