
#ifndef DTYPE_H
#define DTYPE_H

#define NB_DTYPES 10

#include <cstdint>


namespace DType
{
    struct DType
    {
        uint8_t id;
        uint8_t size;
    };

    const DType INT8 = {0, sizeof(int8_t)};
    const DType UINT8 = {1, sizeof(uint8_t)};
    const DType INT16 = {2, sizeof(int16_t)};
    const DType UINT16 = {3, sizeof(uint16_t)};
    const DType INT32 = {4, sizeof(int32_t)};
    const DType UINT32 = {5, sizeof(uint32_t)};
    const DType INT64 = {6, sizeof(int64_t)};
    const DType UINT64 = {7, sizeof(uint64_t)};
    const DType FLOAT32 = {8, sizeof(float)};
    const DType FLOAT64 = {10, sizeof(double)};

    const DType (&DTYPES)[] = {INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64};
};

#endif