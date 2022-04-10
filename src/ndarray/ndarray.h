
#ifndef NDARRAY_CORE_H
#define NDARRAY_CORE_H

#include "src/ndarray/ndarray_types.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>
#include <type_traits>

namespace laruen::ndarray {
    
    template <typename T = float64_t> class NDArray {
        T *data;
        Shape shape;
        Strides strides;
        uint64_t size;
        uint8_t ndim;
        bool free_mem;

        template <typename> friend class NDArray;

        public:
            ~NDArray();
            NDArray();
            NDArray(const Shape &shape);
            NDArray(const Shape &shape, T fill);
            NDArray(T start, T end, T step);
            NDArray(T *data, const Shape &shape, const Strides &strides, uint64_t size, uint8_t ndim, bool free_mem);
            NDArray(const NDArray &ndarray);
            NDArray(NDArray &&ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(NDArray<T2> &&ndarray);

            NDArray& operator=(const NDArray &ndarray);
            NDArray& operator=(NDArray &&ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(NDArray<T2> &&ndarray);

            template <typename T2> void copy_data_from(const NDArray<T2> &ndarray);
            NDArray shallow_copy();
            const NDArray shallow_copy() const;
            void fill(T fill);

            void reshape(const Shape &shape);
            uint64_t ravel_ndindex(const NDIndex &ndindex) const;
            NDIndex unravel_index(uint64_t index) const;
            NDArray shrink_dims() const;
            template <typename T2> bool eq_dims(const NDArray<T2> &ndarray) const;
            T max() const;
            uint64_t index_max() const;
            NDIndex ndindex_max() const;
            T min() const;
            uint64_t index_min() const;
            NDIndex ndindex_min() const;

            std::string info() const;

            T& operator[](const NDIndex &ndindex);
            const T& operator[](const NDIndex &ndindex) const;
            // NDArray operator[](const SliceRanges &slice_ranges);
            // const NDArray operator[](const SliceRanges &slice_ranges) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator+=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator-=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator*=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator/=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator+(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator-(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator*(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator/(T2 value) const;
            template <typename T2> auto operator+(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator-(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator*(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator/(const NDArray<T2> &ndarray) const;
            template <typename T2> NDArray& operator+=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator-=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator*=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator/=(const NDArray<T2> &ndarray);
            template <typename T2> bool operator==(const NDArray<T2> &ndarray) const;
            template <typename T2> bool operator!=(const NDArray<T2> &ndarray) const;
            template <typename T2> bool operator>=(const NDArray<T2> &ndarray) const;
            template <typename T2> bool operator<=(const NDArray<T2> &ndarray) const;
            template <typename T2> bool operator>(const NDArray<T2> &ndarray) const;
            template <typename T2> bool operator<(const NDArray<T2> &ndarray) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator^=(T2 value);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator&=(T2 value);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator|=(T2 value);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator<<=(T2 value);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator>>=(T2 value);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator^(T2 value) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator&(T2 value) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator|(T2 value) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator<<(T2 value) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator>>(T2 value) const;
            NDArray operator~() const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator^=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator&=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator|=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator<<=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> NDArray& operator>>=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator^(const NDArray<T2> &ndarray) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator&(const NDArray<T2> &ndarray) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator|(const NDArray<T2> &ndarray) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator<<(const NDArray<T2> &ndarray) const;
            template <typename T2, typename = std::enable_if_t<types::both_integers_v<T, T2>>> auto operator>>(const NDArray<T2> &ndarray) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2> auto operator%(T2 value) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2> &ndarray);
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2> &ndarray);

        private:
            void str_(std::string &str, uint8_t dim=0, uint64_t data_index=0, bool not_first=false, bool not_last=true) const;
            void shape_array(const Shape &shape);
            void slice_array(const SliceRanges &slice_ranges);
        
        public:
            inline std::string str() const {
                std::string str;
                this->str_(str);
                return str;
            }

            inline const T* get_data() const {
                return this->data;
            }

            inline const Shape& get_shape() const {
                return this->shape;
            }

            inline const Strides& get_strides() const {
                return this->strides;
            }

            inline uint64_t get_size() const {
                return this->size;
            }

            inline uint8_t get_ndim() const {
                return this->ndim;
            }

            inline bool does_free_mem() const {
                return this->free_mem;
            }

            inline void set_free_mem(bool free_mem) {
                this->free_mem = free_mem;
            }

            inline T& operator[](uint64_t index) {
                return this->data[index];
            }

            inline const T& operator[](uint64_t index) const {
                return this->data[index];
            }

            inline friend NDArray operator+(T value, NDArray ndarray) {
                return ndarray + value;
            }

            inline friend NDArray operator-(T value, NDArray ndarray) {
                return ndarray + value;
            }

            inline friend NDArray operator*(T value, NDArray ndarray) {
                return ndarray + value;
            }

            inline friend NDArray operator/(T value, NDArray ndarray) {
                return ndarray + value;
            }
    };
};

#include "src/ndarray/ndarray.tpp"
#endif