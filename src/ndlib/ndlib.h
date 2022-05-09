
#ifndef NDLIB_H
#define NDLIB_H

#include "src/ndlib/ndarray_utils.h"

using namespace laruen;

namespace laruen::ndlib {
    template <auto Op, typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& invoke_broadcast_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <auto Op, typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& invoke_ndarray_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) noexcept;

    template <auto Op, typename T1, bool C1, typename T2>
    NDArray<T1, C1>& invoke_value_assignment(NDArray<T1, C1> &lhs, T2 value) noexcept;


    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& add_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::addition<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::addition<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& subtract_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::subtraction<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::subtraction<T1, T2>>(lhs, rhs);
        }
    }
    
    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& multiply_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::multiplication<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::multiplication<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& divide_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::division<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::division<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& bit_xor_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::bit_xor<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::bit_xor<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& bit_and_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::bit_and<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::bit_and<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& bit_or_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::bit_or<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::bit_or<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& shl_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::bit_shl<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::bit_shl<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& shr_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::bit_shr<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::bit_shr<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& remainder_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::remainder<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::remainder<T1, T2>>(lhs, rhs);
        }
    }

    template <bool B, typename T1, bool C1, typename T2, bool C2>
    inline NDArray<T1, C1>& power_assign(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        if constexpr(B) {
            return invoke_broadcast_assignment<ndlib::utils::operations::power<T1, T2>>(lhs, rhs);
        }
        else {
            return invoke_ndarray_assignment<ndlib::utils::operations::power<T1, T2>>(lhs, rhs);
        }
    }
};

#include "src/ndlib/ndlib.tpp"
#endif