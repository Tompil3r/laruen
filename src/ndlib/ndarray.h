
#ifndef NDARRAY_H
#define NDARRAY_H

#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/nditerator.h"
#include "src/ndlib/array_base.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <type_traits>

namespace laruen::ndlib {
    
    template <typename T = float64_t, bool C = true> class NDArray : public ArrayBase {
        T *m_data;

        template <typename, bool> friend class NDArray;
        friend class NDIterator<T, C>;
        friend class ConstNDIterator<T, C>;

        public:
            ~NDArray();
            NDArray();
            NDArray(const Shape &shape);
            NDArray(const Shape &shape, T value);
            NDArray(T *data, const ArrayBase &base);
            NDArray(T *data, const ArrayBase &base, bool free_mem);
            NDArray(const NDArray &ndarray);
            NDArray(NDArray &&ndarray);
            NDArray(T end);
            NDArray(T start, T end);
            NDArray(T start, T end, T step);
            template <bool C2> NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges);
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(NDArray<T2, C2> &&ndarray);

            NDArray& operator=(const NDArray &ndarray);
            NDArray& operator=(NDArray &&ndarray);
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(NDArray<T2, C2> &&ndarray);

            template <typename T2, bool C2> void copy_data_from(const NDArray<T2, C2> &ndarray);
            void fill(T value);

            T max() const;
            uint64_t index_max() const;
            NDIndex ndindex_max() const;
            T min() const;
            uint64_t index_min() const;
            NDIndex ndindex_min() const;

            T& operator[](const NDIndex &ndindex);
            const T& operator[](const NDIndex &ndindex) const;
            NDArray<T, false> operator[](const SliceRanges &ranges);
            const NDArray<T, false> operator[](const SliceRanges &ranges) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator+=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator-=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator*=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator/=(T2 value);
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator+(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator-(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator*(T2 value) const;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator/(T2 value) const;
            template <typename T2, bool C2> auto operator+(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator-(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator*(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator/(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> NDArray& operator+=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator-=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator*=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator/=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> bool operator==(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> bool operator!=(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> bool operator>=(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> bool operator<=(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> bool operator>(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> bool operator<(const NDArray<T2, C2> &ndarray) const;
            template <typename T2> NDArray& operator^=(T2 value);
            template <typename T2> NDArray& operator&=(T2 value);
            template <typename T2> NDArray& operator|=(T2 value);
            template <typename T2> NDArray& operator<<=(T2 value);
            template <typename T2> NDArray& operator>>=(T2 value);
            template <typename T2> auto operator^(T2 value) const;
            template <typename T2> auto operator&(T2 value) const;
            template <typename T2> auto operator|(T2 value) const;
            template <typename T2> auto operator<<(T2 value) const;
            template <typename T2> auto operator>>(T2 value) const;
            NDArray operator~() const;
            template <typename T2, bool C2> NDArray& operator^=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator&=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator|=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator<<=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator>>=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> auto operator^(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator&(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator|(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator<<(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator>>(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2> auto operator%(T2 value) const;
            template <typename T2, bool C2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> auto operator%(const NDArray<T2, C2> &ndarray) const;

            

        private:
            void str_(std::string &str, uint8_t dim=0, uint64_t data_index=0, bool not_first=false, bool not_last=true) const;
        
        public:
            inline std::string str() const {
                std::string str;
                this->str_(str);
                return str;
            }

            inline const T* data() const {
                return this->m_data;
            }

            inline T& operator[](uint64_t index) {
                return this->m_data[index];
            }

            inline const T& operator[](uint64_t index) const {
                return this->m_data[index];
            }
    };
};

#include "src/ndlib/ndarray.tpp"
#endif