
#ifndef NDARRAY_STATIC_H
#define NDARRAY_STATIC_H

#include "src/ndlib/ndarray.h"

class NDArrayStatic {

    public:
        template <typename T, bool C, typename T2, bool C2>
        static NDArray<T, C>& add_assign_normal(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs);

        template <typename T, bool C, typename T2, bool C2>
        static NDArray<T, C>& add_assign_broadcast(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs);
};

#include "src/ndlib/ndarray_static.tpp"
#endif