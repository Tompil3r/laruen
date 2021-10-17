
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/typenames.h"
#include <vector>
#include <cstdint>
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


        void print(uint8_t dim = 0, uint64_t data_index = 0, bool not_first = false, bool not_last = true) const
        {
            uint32_t dim_idx;

            if(not_first) std::cout << std::string(dim, ' '); 
            std::cout << '[';

            if(dim == this->ndim - 1)
            {
                for(dim_idx = 0;dim_idx < this->shape[dim] - 1;dim_idx++)
                {
                    std::cout << this->data[data_index + dim_idx] << ", ";
                }

                std::cout << this->data[data_index + dim_idx] << ']';
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