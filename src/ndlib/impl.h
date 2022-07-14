#ifndef NDLIB_STATIC_H_
#define NDLIB_STATIC_H_

#include <cstdint>
#include "src/math/common.h"

namespace laruen::ndlib::impl {

    template <typename TR, typename T, typename T2>
    inline TR dot_1d(T *lhs_ptr, uint_fast64_t lhs_stride,
    T2 *rhs_ptr, uint_fast64_t rhs_stride, uint_fast64_t size) noexcept {
        TR product = 0;
        for(uint_fast64_t i = 0;i < size;i++) {
            product += (*lhs_ptr) * (*rhs_ptr);
            lhs_ptr += lhs_stride;
            rhs_ptr += rhs_stride;
        }

        return product;
    }

    template <typename T, typename T2, typename TR>
    TR* matmul_2d_n3(T *lhs_ptr, uint_fast64_t lhs_row_stride, uint_fast64_t lhs_col_stride,
    T2 *rhs_ptr, uint_fast64_t rhs_row_stride, uint_fast64_t rhs_col_stride,
    TR *out_ptr, uint_fast64_t out_row_stride, uint_fast64_t out_col_stride,
    uint_fast64_t rows, uint_fast64_t cols, uint_fast64_t shared)
    {
        T2 *rhs_start_ptr = rhs_ptr;
        TR *out_start_ptr = out_ptr;
        TR *out_checkpoint = out_ptr;

        for(uint_fast64_t row = 0;row < rows;row++) {
            for(uint_fast64_t col = 0;col < cols;col++) {
                *out_ptr = dot_1d<TR>(lhs_ptr, lhs_col_stride, rhs_ptr, rhs_row_stride, shared);
                out_ptr += out_col_stride;
                rhs_ptr += rhs_col_stride;
            }
            out_checkpoint += out_row_stride;
            out_ptr = out_checkpoint;
            lhs_ptr += lhs_row_stride;
            rhs_ptr = rhs_start_ptr;
        }

        return out_start_ptr;
    }

    template <typename T, typename T2>
    T* add_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() += rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* add_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() += value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* add(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() + value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* add(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() + rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* subtract_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() -= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* subtract_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() -= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* subtract(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() - value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* subtract(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() - rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* multiply_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() *= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* multiply_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() *= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* multiply(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() * value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* multiply(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() * rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* divide_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() /= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* divide_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() /= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* divide(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() / value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* divide(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */

        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() / rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* bit_xor_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() ^= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* bit_xor_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() ^= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* bit_xor(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() ^ value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* bit_xor(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() ^ rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* bit_and_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() &= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* bit_and_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() &= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* bit_and(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() & value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* bit_and(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() & rhs_iter.next();
        }

        return out_data;
    }
    
    template <typename T, typename T2>
    T* bit_or_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() |= rhs_iter.next();
        }

        return lhs_data;
	}
    
	template <typename T>
    T* bit_or_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() |= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* bit_or(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() | value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* bit_or(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() | rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* shl_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() <<= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* shl_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() <<= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* shl(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() << value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* shl(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() << rhs_iter.next();
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* shr_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() >>= rhs_iter.next();
        }

        return lhs_data;
	}

	template <typename T>
    T* shr_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() >>= value;
        }

        return data;
	}

    template <typename T, typename TR>
    TR* shr(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() >> value;
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* shr(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = lhs_iter.next() >> rhs_iter.next();
        }

        return out_data;
    }

    template <typename T>
    T* bit_not_eq(T *data, const ArrayBase &base) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() = ~iter.current();
        }

        return data;
    }

    template <typename T, typename TR>
    TR* bit_not(const T *lhs_data, const ArrayBase &lhs_base, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint_fast64_t i = 0 ;i < lhs_base.size();i++) {
            out_iter.next() = ~lhs_iter.next();
        }
    }

    template <typename T, typename T2>
    T* remainder_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() = laruen::math::common::remainder(lhs_iter.current(), rhs_iter.next());
        }

        return lhs_data;
	}

	template <typename T>
    T* remainder_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() = laruen::math::common::remainder(iter.current(), value);
        }

        return data;
	}

    template <typename T, typename TR>
    TR* remainder(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), value);
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* remainder(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = laruen::math::common::remainder(lhs_iter.next(), rhs_iter.next());
        }

        return out_data;
    }

    template <typename T, typename T2>
    T* power_eq(T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base) {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);

        for(uint_fast64_t i = 0;i < lhs_base.size();i++) {
            lhs_iter.next() = laruen::math::common::pow(lhs_iter.current(), rhs_iter.next());
        }

        return lhs_data;
	}

	template <typename T>
    T* power_eq(T *data, const ArrayBase &base, T value) noexcept {
        NDIter iter(data, base);

        for(uint_fast64_t i = 0;i < base.size();i++) {
            iter.next() = laruen::math::common::pow(iter.current(), value);
        }

        return data;
	}

    template <typename T, typename TR>
    TR* power(const T *lhs_data, const ArrayBase &lhs_base, TR value, TR *out_data, const ArrayBase &out_base) noexcept {
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = laruen::math::common::pow(lhs_iter.next(), value);
        }
        
        return out_data;
    }

    template <typename T, typename T2, typename TR>
    TR* power(const T *lhs_data, const ArrayBase &lhs_base, const T2 *rhs_data, const ArrayBase &rhs_base, TR *out_data, const ArrayBase &out_base) {
        /* implementation function: arrays must be broadcasted if needed */
        
        NDIter lhs_iter(lhs_data, lhs_base);
        NDIter rhs_iter(rhs_data, rhs_base);
        NDIter out_iter(out_data, out_base);

        for(uint64_t i = 0;i < lhs_base.size();i++) {
            out_iter.next() = laruen::math::common::pow(lhs_iter.next(), rhs_iter.next());
        }

        return out_data;
    }
}

#endif