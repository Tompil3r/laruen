
#ifndef NDLIB_TYPE_SELECTION_H_
#define NDLIB_TYPE_SELECTION_H_

#include <cstdint>
#include <type_traits>
#include "src/ndlib/ndarray.h"

namespace laruen::ndlib {
    // NDArray forward declaration
    template <typename T, bool C> class NDArray;

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

        template <typename T, typename T2>
        struct max_type {
            typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<T2>,
                std::conditional_t<sizeof(T) >= sizeof(T2), T, T2>,
                std::conditional_t<std::is_floating_point_v<T>, T, T2>> type;
        };
        template <typename T, typename T2> using max_type_t = typename max_type<T, T2>::type;

        template <typename T, typename T2> struct float_type {
            typedef std::conditional_t<std::is_floating_point_v<T>, T, T2> type;
        };
        template <typename T, typename T2> using float_type_t = typename float_type<T, T2>::type;

        template <typename T, typename T2> struct integer_type {
            typedef std::conditional_t<std::is_integral_v<T>, T, T2> type;
        };
        template <typename T, typename T2> using integer_type_t = typename integer_type<T, T2>::type;

        template <typename T, typename T2>
        struct combine_types {
        /*
            code simplification:
            if(is_int(T) == is_int(T2)) {
                if(is_signed(T) == is_signed(T2) || (is_signed(max_type(T, T2)) && sizeof(T) != sizeof(T2))) {
                    type = max_type(T, T2);
                }
                else {
                    type = next_signed(max(T, T2));
                }
            }
            else {
                if(sizeof(int_type(T, T2)) >= sizeof(float_type(T, T2))) {
                    type = next_signed(max_type(T, T2));
                }   
                else {
                    type = max_type(T, T2)
                }
            }
        */
        typedef std::conditional_t<std::is_integral_v<T> == std::is_integral_v<T2>,
            // group a - both ints or both floats
            std::conditional_t<std::is_signed_v<T> == std::is_signed_v<T2> ||
                (std::is_signed_v<max_type_t<T, T2>> && sizeof(T) != sizeof(T2)),
                // sub group a1 - (both signed or both unsigned) or (one signed and one unsigned and have different sizes)
                max_type_t<T, T2>,
                // sub group a2 - one signed and one unsigned (order does not matter) and max or equal size is unsigned
                next_signed_t<max_type_t<T, T2>>>,

            // group b - one int and one float (order does not matter)
            std::conditional_t<sizeof(integer_type_t<T, T2>) >= sizeof(float_type_t<T, T2>),
                // sub group b1 - the size of the integer type is bigger or equal to the size of the float type
                next_signed_t<max_type_t<T, T2>>,
                // sub group b2 - the size of the integer type is smaller than the size of the float type
                max_type_t<T, T2>>> type;
        };
        template <typename T, typename T2> using combine_types_t = typename combine_types<T, T2>::type;

        template <typename T> struct is_ndarray {
            static constexpr bool value = false;
        };
        template <typename T> inline constexpr bool is_ndarray_v = is_ndarray<T>::value;

        template <typename T, bool C> struct is_ndarray<NDArray<T, C>> {
            static constexpr bool value = true;
        };

        template <typename T, bool C> struct is_ndarray<const NDArray<T, C>> {
            static constexpr bool value = true;
        };

        template <typename T, typename T2> inline constexpr bool both_integers_v = (std::is_integral_v<T> && std::is_integral_v<T2>);
        template <typename T, typename T2> inline constexpr bool atleast_one_float_v = (std::is_floating_point_v<T> || std::is_floating_point_v<T2>);
    }
}

#endif