
#include "src/ndlib/ndlib.h"
#include "src/ndlib/ndarray_types.h"
#include "src/ndlib/ndarray_utils.h"
#include "src/math/common.h"
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

using namespace laruen;

namespace laruen::ndlib {

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& add_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() += rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& add_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
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
    
    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& subtract_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() -= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& subtract_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() -= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& multiply_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() *= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& multiply_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() *= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& divide_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() /= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& divide_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() /= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& xor_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() ^= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& xor_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() ^= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& and_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() &= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& and_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() &= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& or_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() |= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& or_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() |= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shl_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() <<= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shl_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() <<= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shr_assign_normal(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDIterator lhs_iter(lhs);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < lhs.m_size;i++) {
            lhs_iter.next() >>= rhs_iter.next();
        }

        return lhs;
    }

    template <typename T1, bool C1, typename T2, bool C2>
    NDArray<T1, C1>& shr_assign_broadcast(NDArray<T1, C1> &lhs, const NDArray<T2, C2> &rhs) {
        NDArray<T1, false> reorder = ndlib::utils::broadcast_reorder(lhs, rhs);
        uint64_t size_ratio = lhs.m_size / rhs.m_size;
        NDIterator lhs_iter(reorder);
        ConstNDIterator rhs_iter(rhs);

        for(uint64_t i = 0;i < size_ratio;i++) {
            for(uint64_t j = 0;j < rhs.m_size;j++) {
                lhs_iter.next() >>= rhs_iter.next();
            }
            rhs_iter.reset();
        }

        return lhs;
    }
}