
#ifndef NDARRAY_H
#define NDARRAY_H

#include "src/ndlib/ndarray_utils.h"
#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/nditerator.h"
#include "src/ndlib/array_base.h"
#include "src/utils/range.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <type_traits>

using laruen::utils::Range;

namespace laruen::ndlib {

    template <typename T = float64_t, bool C = true> class NDArray : public ArrayBase {
        template <typename, bool> friend class NDArray;
        friend class NDIterator<T, C>;
        friend class ConstNDIterator<T, C>;

        private:
            T *m_data;

        public:
            ~NDArray();
            NDArray() noexcept;
            NDArray(T *data, const Shape &shape, const Strides &strides,
            uint64_t size, uint8_t ndim, bool free_mem) noexcept;
            NDArray(T *data, Shape &&shape, Strides &&strides,
            uint64_t size, uint8_t ndim, bool free_mem) noexcept;
            NDArray(const Shape &shape) noexcept;
            NDArray(const Shape &shape, T value) noexcept;
            NDArray(T *data, const ArrayBase &base) noexcept;
            NDArray(T *data, const ArrayBase &base, bool free_mem) noexcept;
            NDArray(const NDArray &ndarray) noexcept;
            NDArray(NDArray &&ndarray) noexcept;
            NDArray(const Range<T> &range) noexcept;
            NDArray(NDArray &ndarray, const Axes &axes) noexcept;
            template <bool C2> NDArray(NDArray<T, C2> &ndarray, const SliceRanges &ranges) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2> || C != C2>> NDArray(const NDArray<T2, C2> &ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2> || C != C2>> NDArray(NDArray<T2, C2> &&ndarray) noexcept;

            NDArray& operator=(const NDArray &ndarray) noexcept;
            NDArray& operator=(NDArray &&ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(const NDArray<T2, C2> &ndarray) noexcept;
            template <typename T2, bool C2, typename = std::enable_if_t<!std::is_same_v<T, T2>>> NDArray& operator=(NDArray<T2, C2> &&ndarray) noexcept;

            template <typename T2, bool C2> void copy_data_from(const NDArray<T2, C2> &ndarray) noexcept;
            void fill(T value) noexcept;

            T max() const noexcept;
            uint64_t index_max() const noexcept;
            NDIndex ndindex_max() const noexcept;
            T min() const noexcept;
            uint64_t index_min() const noexcept;
            NDIndex ndindex_min() const noexcept;

            T& operator[](const NDIndex &ndindex) noexcept;
            const T& operator[](const NDIndex &ndindex) const noexcept;
            NDArray<T, false> operator[](const SliceRanges &ranges) noexcept;
            const NDArray<T, false> operator[](const SliceRanges &ranges) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator+=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator-=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator*=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> NDArray& operator/=(T2 value) noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator+(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator-(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator*(T2 value) const noexcept;
            template <typename T2, typename = std::enable_if_t<!types::is_ndarray_v<T2>>> auto operator/(T2 value) const noexcept;
            template <typename T2, bool C2> auto operator+(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator-(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator*(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> auto operator/(const NDArray<T2, C2> &ndarray) const;
            template <typename T2, bool C2> NDArray& operator+=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator-=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator*=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> NDArray& operator/=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> bool operator==(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator!=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator>=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator<=(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator>(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2, bool C2> bool operator<(const NDArray<T2, C2> &ndarray) const noexcept;
            template <typename T2> NDArray& operator^=(T2 value) noexcept;
            template <typename T2> NDArray& operator&=(T2 value) noexcept;
            template <typename T2> NDArray& operator|=(T2 value) noexcept;
            template <typename T2> NDArray& operator<<=(T2 value) noexcept;
            template <typename T2> NDArray& operator>>=(T2 value) noexcept;
            template <typename T2> auto operator^(T2 value) const noexcept;
            template <typename T2> auto operator&(T2 value) const noexcept;
            template <typename T2> auto operator|(T2 value) const noexcept;
            template <typename T2> auto operator<<(T2 value) const noexcept;
            template <typename T2> auto operator>>(T2 value) const noexcept;
            NDArray operator~() const noexcept;
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
            template <typename T2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value) noexcept;
            template <typename T2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(T2 value) noexcept;
            template <typename T2> auto operator%(T2 value) const noexcept;
            template <typename T2, bool C2, std::enable_if_t<!types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2, std::enable_if_t<types::atleast_one_float_v<T, T2>, int> = 0> NDArray& operator%=(const NDArray<T2, C2> &ndarray);
            template <typename T2, bool C2> auto operator%(const NDArray<T2, C2> &ndarray) const;

            

        private:
            void str_(std::string &str, uint8_t dim=0, uint64_t data_index=0, bool not_first=false, bool not_last=true) const noexcept;
        
        public:
            inline std::string str() const noexcept {
                std::string str;
                this->str_(str);
                return str;
            }

            inline const T* data() const noexcept {
                return this->m_data;
            }

            inline T& operator[](uint64_t index) noexcept {
                return this->m_data[index];
            }

            inline const T& operator[](uint64_t index) const noexcept {
                return this->m_data[index];
            }

            /* ----- ndlib -----  */
            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, false> ndlib::utils::broadcast_reorder(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& add_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& add_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& subtract_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& subtract_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& multiply_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& multiply_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& divide_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

            template <typename T1, bool C1, typename T2, bool C2>
            friend NDArray<T1, C1>& divide_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);
    };
    
    template <typename T, bool C> NDArray(NDArray<T, C>&, const Axes&) -> NDArray<T, false>;
};

#include "src/ndlib/ndarray.tpp"
#endif