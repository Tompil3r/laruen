
#ifndef DTYPE_H
#define DTYPE_H

#define NB_DTYPES 10

#include <cstdint>


namespace DType
{
    struct DType
    {
        bool dtype_signed;
        uint8_t size;
    };

    const DType INT8 = {true, sizeof(int8_t)};
    const DType UINT8 = {false, sizeof(uint8_t)};
    const DType INT16 = {true, sizeof(int16_t)};
    const DType UINT16 = {false, sizeof(uint16_t)};
    const DType INT32 = {true, sizeof(int32_t)};
    const DType UINT32 = {false, sizeof(uint32_t)};
    const DType INT64 = {true, sizeof(int64_t)};
    const DType UINT64 = {false, sizeof(uint64_t)};
    const DType FLOAT32 = {true, sizeof(float)};
    const DType FLOAT64 = {true, sizeof(double)};

    const DType (&DTYPES)[] = {INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64};
};

#endif