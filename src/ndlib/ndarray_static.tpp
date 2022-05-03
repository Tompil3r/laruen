
#include "src/ndlib/ndarray_static.h"
#include "src/ndlib/nditerator.h"
#include "src/ndlib/ndarray.h"
#include "src/ndlib/ndarray_utils.h"

using namespace laruen;
using laruen::ndlib::NDArray;
using laruen::ndlib::NDIterator;
using laruen::ndlib::ConstNDIterator;

template <typename T, bool C, typename T2, bool C2>
NDArray<T, C>& NDArrayStatic::add_assign_normal(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs) {
    NDIterator lhs_iter(lhs);
    ConstNDIterator rhs_iter(rhs);

    for(uint64_t i = 0;i < lhs.m_size;i++) {
        lhs_iter.next() += rhs_iter.next();
    }

    return lhs;
}

template <typename T, bool C, typename T2, bool C2>
NDArray<T, C>& NDArrayStatic::add_assign_broadcast(NDArray<T, C> &lhs, const NDArray<T2, C2> &rhs) {
    NDArray<T, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
    uint64_t size_ratio = lhs.m_size / rhs.m_size;
    NDIterator lhs_iter(reorder);
    ConstNDIterator rhs_iter(rhs);

    for(uint64_t i = 0;i < size_ratio;i++) {
        for(uint64_t j = 0;j < rhs.m_size;j++) {
            lhs_iter.next() += rhs_iter.next();
        }
        rhs_iter.reset();
    }

    return lhs;
}