
#ifndef NDLIB_H
#define NDLIB_H

namespace laruen::ndlib {
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
};

#include "src/ndlib/ndlib.tpp"
#endif