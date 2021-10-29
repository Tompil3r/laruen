
#ifndef NDARRAY_H
#define NDARRAY_H

#include "laruen/ndarray/typenames.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>

namespace laruen::ndarray
{
    template <typename T> class NDArray
    {
        T *data;
        Shape shape;
        Strides strides;
        uint8_t ndim;
        uint64_t size;
        bool delete_data;
        

        public:
            ~NDArray();
            NDArray();
            NDArray(const Shape &shape);
            NDArray(const Shape &shape, T fill_value);
            NDArray(T start, T stop, T step);
            NDArray(T *data, const Shape &shape, const Strides &strides, uint8_t ndim, uint64_t size, bool delete_data);
            NDArray(const NDArray &ndarray);

            const T* get_data() const;
            const Shape& get_shape() const;
            const Strides& get_strides() const;
            uint8_t get_ndim() const;
            uint64_t get_size() const;
            bool does_delete_data();
            void set_delete_data(bool delete_date);

            void reshape(const Shape &shape);
            uint64_t ravel_ndindex(const NDIndex &ndindex) const;
            NDIndex unravel_index(uint64_t index) const;

            std::string get_specs() const;
            void print(bool specs=false, uint8_t dim=0, uint64_t data_index=0, bool not_first=false, bool not_last=true) const;

            T& operator[](uint64_t index);
            const T& operator[](uint64_t index) const;
            T& operator[](const NDIndex &ndindex);
            const T& operator[](const NDIndex &ndindex) const;
        
        private:
            void construct_shape(const Shape &shape);
    };
};

#endif