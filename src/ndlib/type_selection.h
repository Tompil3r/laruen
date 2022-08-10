
#ifndef NDLIB_TYPE_SELECTION_H_
#define NDLIB_TYPE_SELECTION_H_

#include <cstdint>
#include <type_traits>
#include "src/ndlib/ndarray.h"

namespace laruen::ndlib {
    // NDArray forward declaration
    template <typename T> class NDArray;

    namespace types {
        template <typename T> struct next_signed;
        template <> struct next_signed<int8_t> { typedef int16_t type; };
        template <> struct next_signed<uint8_t> { typedef int16_t type; };
        template <> struct next_signed<int16_t> { typedef int32_t type; };
        template <> struct next_signed<uint16_t> { typedef int32_t type; };
        template <> struct next_signed<int32_t> { typedef int64_t type; };
        template <> struct next_signed<uint32_t> { typedef int64_t type; };
        template <> struct next_signed<int64_t> { typedef float64_t type; };
        template <> struct next_signed<uint64_t> { typedef float64_t type; };
        template <> struct next_signed<float32_t> { typedef float64_t type; };
        template <> struct next_signed<float64_t> { typedef float64_t type; };
        template <typename T> using next_signed_t = typename next_signed<T>::type;

        template <typename T, typename TT>
        struct max_type {
            typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<TT>,
                std::conditional_t<sizeof(T) >= sizeof(TT), T, TT>,
                std::conditional_t<std::is_floating_point_v<T>, T, TT>> type;
        };
        template <typename T, typename TT> using max_type_t = typename max_type<T, TT>::type;

        template <typename T, typename TT> struct float_type {
            typedef std::conditional_t<std::is_floating_point_v<T>, T, TT> type;
        };
        template <typename T, typename TT> using float_type_t = typename float_type<T, TT>::type;

        template <typename T, typename TT> struct integer_type {
            typedef std::conditional_t<std::is_integral_v<T>, T, TT> type;
        };
        template <typename T, typename TT> using integer_type_t = typename integer_type<T, TT>::type;

        template <typename T, typename TT>
        struct result_type {
        /*
            code simplification:
            if(is_int(T) == is_int(TT)) {
                if(is_signed(T) == is_signed(TT) || (is_signed(max_type(T, TT)) && sizeof(T) != sizeof(TT))) {
                    type = max_type(T, TT);
                }
                else {
                    type = next_signed(max(T, TT));
                }
            }
            else {
                if(sizeof(int_type(T, TT)) >= sizeof(float_type(T, TT))) {
                    type = next_signed(max_type(T, TT));
                }   
                else {
                    type = max_type(T, TT)
                }
            }
        */
        typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<TT>,
            // group a - both ints or both floats
            std::conditional_t<std::is_signed_v<T> == std::is_signed_v<TT> ||
                (std::is_signed_v<max_type_t<T, TT>> && sizeof(T) != sizeof(TT)),
                // sub group a1 - (both signed or both unsigned) or (one signed and one unsigned and have different sizes)
                max_type_t<T, TT>,
                // sub group a2 - one signed and one unsigned (order does not matter) and max or equal size is unsigned
                next_signed_t<max_type_t<T, TT>>>,

            // group b - one int and one float (order does not matter)
            std::conditional_t<sizeof(integer_type_t<T, TT>) >= sizeof(float_type_t<T, TT>),
                // sub group b1 - the size of the integer type is bigger or equal to the size of the float type
                next_signed_t<max_type_t<T, TT>>,
                // sub group b2 - the size of the integer type is smaller than the size of the float type
                max_type_t<T, TT>>> type;
        };
        template <typename T, typename TT> using result_type_t = typename result_type<T, TT>::type;

        template <typename T> struct is_ndarray {
            static constexpr bool value = false;
        };
        template <typename T> inline constexpr bool is_ndarray_v = is_ndarray<T>::value;

        template <typename T> struct is_ndarray<NDArray<T>> {
            static constexpr bool value = true;
        };

        template <typename T> struct is_ndarray<const NDArray<T>> {
            static constexpr bool value = true;
        };
    }
}

#endif