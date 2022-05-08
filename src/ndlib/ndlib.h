
#ifndef NDLIB_H
#define NDLIB_H

namespace laruen::ndlib {
    template <auto Op, typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& invoke_broadcast_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <auto Op, typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& invoke_ndarray_assignment(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) noexcept;

    template <auto Op, typename T1, bool C1, typename T2>
    NDArray<T1, C1>& invoke_value_assignment(NDArray<T1, C1> &lhs, T2 value) noexcept;

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& add_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& add_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& subtract_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& subtract_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& multiply_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& multiply_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& divide_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& divide_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& xor_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& xor_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& and_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& and_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& or_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& or_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shl_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shl_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shr_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shr_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& remainder_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& remainder_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs);
};

#include "src/ndlib/ndlib.tpp"
#endif