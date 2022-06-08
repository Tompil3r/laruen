
#ifndef BITS_H
#define BITS_H

#include <cassert>

namespace laruen::math::bits {

    #if defined(__GNUC__)  // GCC, Clang, ICC

        inline auto lsb(uint64_t n) {
            assert(n);
            return __builtin_ctzll(n);
        }

        inline auto lsb(uint32_t n) {
            assert(n);
            return __builtin_ctz(n);
        }

        inline auto msb(uint64_t n) {
            assert(n);
            return ((decltype(__builtin_clzll(n)))(63) ^ __builtin_clzll(n));
        }

        inline auto msb(uint32_t n) {
            assert(n);
            return ((decltype(__builtin_clz(n)))(31) ^ __builtin_clz(n));
        }

    #elif defined(_MSC_VER)  // MSVC

        // 32 bit versions for both WIN32 and WIN64
        inline unsigned long lsb(uint32_t n) {
            assert(n);
            unsigned long index;
            _BitScanForward(&index, n);
            return index;
        }

        inline unsigned long msb(uint32_t n) {
            assert(n);
            unsigned long index;
            _BitScanReverse(&idx, n);
            return index;
        }

        #ifdef _WIN64  // MSVC, WIN64

            inline unsigned long lsb(uint64_t n) {
                assert(n);
                unsigned long index;
                _BitScanForward64(&index, n);
                return index;
            }

            inline unsigned long msb(uint64_t n) {
                assert(n);
                unsigned long index;
                _BitScanReverse64(&idx, n);
                return index;
            }

        #else  // MSVC, WIN32

            inline unsigned long lsb(uint64_t n) {
                assert(n);
                unsigned long index;
                if(n & 0xFFFFFFFF) {
                    _BitScanForward(&index, (uint32_t)(n));
                    return index;
                }
                else {
                    _BitScanForward(&index, (uint32_t)(n >> 32));
                    return (index + 32);
                }
            }

            inline unsigned long msb(uint64_t n) {
                assert(n);
                unsigned long index;
                if(n >> 32) {
                    _BitScanReverse(&index, (uint32_t)(n >> 32));
                    return (index + 32);
                }
                else {
                    _BitScanReverse(&index, (uint32_t)(n));
                    return index;
                }
            }

        #endif

    #else  // Compiler is neither GCC nor MSVC compatible

        inline uint_fast8_t lsb(uint64_t n) noexcept {
            constexpr uint_fast8_t debruijn_bitposition[] = {
                0, 1, 17, 2, 18, 50, 3, 57,
                47, 19, 22, 51, 29, 4, 33, 58,
                15, 48, 20, 27, 25, 23, 52, 41,
                54, 30, 38, 5, 43, 34, 59, 8,
                63, 16, 49, 56, 46, 21, 28, 32,
                14, 26, 24, 40, 53, 37, 42, 7,
                62, 55, 45, 31, 13, 39, 36, 6,
                61, 44, 12, 35, 60, 11, 10, 9
            };

            assert(n);
            return debruijn_bitposition[((n & (-n)) * 0x37E84A99DAE458FULL) >> 58];
        }
        
        inline uint_fast8_t lsb(uint32_t n) noexcept {
            constexpr uint_fast8_t debruijn_bitposition[] = {
                0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
                31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
            };

            assert(n);
            return debruijn_bitposition[((n & (-n)) * 0x077CB531U) >> 27];
        }

        inline uint_fast8_t msb(uint64_t n) noexcept {
            constexpr uint8_t debruijn_bitposition[] = {
                0, 47,  1, 56, 48, 27,  2, 60,
                57, 49, 41, 37, 28, 16,  3, 61,
                54, 58, 35, 52, 50, 42, 21, 44,
                38, 32, 29, 23, 17, 11,  4, 62,
                46, 55, 26, 59, 40, 36, 15, 53,
                34, 51, 20, 43, 31, 22, 10, 45,
                25, 39, 14, 33, 19, 30,  9, 24,
                13, 18,  8, 12,  7,  6,  5, 63
            };

            assert(n);
            n |= n >> 1; 
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n |= n >> 32;
            return deburijn_bitposition[(n * 0x03F79D71B4CB0A89ULL) >> 58];
        }

        inline uint_fast8_t msb(uint32_t n) noexcept {
            constexpr uint_fast8_t debruijn_bitposition[] = {
                0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
                8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
            };

            assert(n);
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return debruijn_bitposition[(n * 0x07C4ACDDU) >> 27];
        }

    #endif
};

#endif