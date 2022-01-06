
#ifndef NDARRAY_CORE_H
#define NDARRAY_CORE_H

#include "laruen/ndarray/ndarray_types.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>

namespace laruen::ndarray
{
    template <typename T, uint8_t NDim> class NDArray
    {
        T *data;
        Shape shape;
        Strides strides;
        uint64_t size;
        bool delete_data;

        template <typename, uint8_t> friend class NDArray;

        public:
            NDArray& operator=(const NDArray &ndarray);
            NDArray& operator=(NDArray &&ndarray);
            ~NDArray();
            NDArray();
            NDArray(const Shape &shape);
            NDArray(const Shape &shape, T fill_value);
            NDArray(T start, T stop, T step);
            NDArray(T *data, const Shape &shape, const Strides &strides, uint64_t size, bool delete_data);
            NDArray(const NDArray &ndarray);
            NDArray(NDArray &&ndarray);

            const T* get_data() const;
            const Shape& get_shape() const;
            const Strides& get_strides() const;
            uint64_t get_size() const;
            bool does_delete_data();
            void set_delete_data(bool delete_date);
            NDArray shallow_copy();
            const NDArray shallow_copy() const;
            void fill(T fill_value);

            template <uint8_t NNDim> NDArray<T, NNDim> reshape(const Shape &shape) const;
            uint64_t ravel_ndindex(const NDIndex &ndindex) const;
            NDIndex unravel_index(uint64_t index) const;
            template <uint8_t NNDim> NDArray<T, NNDim> shrink_dims() const;
            bool dims_equal(const NDArray &ndarray) const;
            T max() const;
            uint64_t index_max() const;
            NDIndex ndindex_max() const;
            T min() const;
            uint64_t index_min() const;
            NDIndex ndindex_min() const;

            std::string get_specs() const;

            T& operator[](uint64_t index);
            const T& operator[](uint64_t index) const;
            T& operator[](const NDIndex &ndindex);
            const T& operator[](const NDIndex &ndindex) const;
            NDArray operator[](const SliceRanges &slice_ranges);
            void operator+=(T value);
            void operator-=(T value);
            void operator*=(T value);
            void operator/=(T value);
            NDArray operator+(T value) const;
            NDArray operator-(T value) const;
            NDArray operator*(T value) const;
            NDArray operator/(T value) const;
            void operator+=(const NDArray &ndarray);
            void operator-=(const NDArray &ndarray);
            void operator*=(const NDArray &ndarray);
            void operator/=(const NDArray &ndarray);
            NDArray operator+(const NDArray &ndarray) const;
            NDArray operator-(const NDArray &ndarray) const;
            NDArray operator*(const NDArray &ndarray) const;
            NDArray operator/(const NDArray &ndarray) const;
            bool operator==(const NDArray &ndarray) const;
            bool operator!=(const NDArray &ndarray) const;
            bool operator>=(const NDArray &ndarray) const;
            bool operator<=(const NDArray &ndarray) const;
            bool operator>(const NDArray &ndarray) const;
            bool operator<(const NDArray &ndarray) const;

        private:
            void print(bool print_specs, uint8_t dim, uint64_t data_index=0, bool not_first=false, bool not_last=true) const;
            void shape_array(const Shape &shape);
            void slice_array(const SliceRanges &slice_ranges);
            // Shape broadcast_shapes(const NDArray &ndarray) const;
            // bool output_broadcastable(const NDArray &ndarray) const;
        
        public:
            inline void print(bool print_specs=false) const
            {
                this->print(print_specs, 0);
            }

            inline friend NDArray operator+(T value, NDArray ndarray)
            {
                return ndarray + value;
            }

            inline friend NDArray operator-(T value, NDArray ndarray)
            {
                return ndarray + value;
            }

            inline friend NDArray operator*(T value, NDArray ndarray)
            {
                return ndarray + value;
            }

            inline friend NDArray operator/(T value, NDArray ndarray)
            {
                return ndarray + value;
            }
    };
};

#include "laruen/ndarray/ndarray_core.tpp"
#endif