
#ifndef NDARRAY_H
#define NDARRAY_H

#include "src/ndarray/ndarray_types.h"
#include "src/ndarray/nditerator.h"
#include "src/ndarray/array_base.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <type_traits>

namespace laruen::ndarray {
    
    template <typename T = float64_t> class NDArray : public ArrayBase {
        T *m_data;

        template <typename> friend class NDArray;
        friend class NDIterator<NDArray<T>>;
        friend class NDIterator<const NDArray<T>>;

        public:
            ~NDArray();
            NDArray();
            NDArray(const Shape &shape);
            NDArray(const Shape &shape, T fill);
            NDArray(T *data, const ArrayBase &base);
            NDArray(T *data, const ArrayBase &base, bool free_mem);
            NDArray(const NDArray &ndarray);
            NDArray(NDArray &&ndarray);
            NDArray(T end);
            NDArray(T start, T end);
            NDArray(T start, T end, T step);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray(NDArray<T2> &&ndarray);

            NDArray& operator=(const NDArray &ndarray);
            NDArray& operator=(NDArray &&ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(const NDArray<T2> &ndarray);
            template <typename T2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(NDArray<T2> &&ndarray);

            template <typename T2> void copy_data_from(const NDArray<T2> &ndarray);
            void fill(T fill);

            T max() const;
            uint64_t index_max() const;
            NDIndex ndindex_max() const;
            T min() const;
            uint64_t index_min() const;
            NDIndex ndindex_min() const;

            T& operator[](const NDIndex &ndindex);
            const T& operator[](const NDIndex &ndindex) const;
            NDArray operator[](const SliceRanges &slice_ranges);
            const NDArray operator[](const SliceRanges &slice_ranges) const;
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
            template <typename T2> NDArray& operator^=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator&=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator|=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator<<=(const NDArray<T2> &ndarray);
            template <typename T2> NDArray& operator>>=(const NDArray<T2> &ndarray);
            template <typename T2> auto operator^(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator&(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator|(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator<<(const NDArray<T2> &ndarray) const;
            template <typename T2> auto operator>>(const NDArray<T2> &ndarray) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value);
            template <typename T2> auto operator%(T2 value) const;
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2> &ndarray);
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2> &ndarray);
            template <typename T2> auto operator%(const NDArray<T2> &ndarray) const;

            

        private:
            void str_(std::string &str, uint8_t dim=0, uint64_t data_index=0, bool not_first=false, bool not_last=true) const;
            void slice_array(const SliceRanges &slice_ranges);
        
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

#include "src/ndarray/ndarray.tpp"
#endif