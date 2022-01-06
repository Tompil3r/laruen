
#include "laruen/utils/array.h"
#include <cstdint>
#include <initializer_list>
#include <cassert>

using namespace laruen::utils;

template <typename T, uint64_t N>
Array<T, N>::~Array()
{
    delete[] this->data;
}

template <typename T, uint64_t N>
Array<T, N>::Array() : data(new T[N])
{}

template <typename T, uint64_t N>
Array<T, N>::Array(T value) : data(new T[N])
{
    for(uint64_t idx = 0;idx < N;idx++)
    {
        this->data[idx] = value;
    }
}

template <typename T, uint64_t N>
Array<T, N>::Array(const std::initializer_list<T> &init_list) : data(new T[N])
{
    assert(init_list.size() == N);

    uint64_t idx = 0;

    for(T value : init_list)
    {
        this->data[idx] = value;
        idx++;
    }
}

template <typename T, uint64_t N>
Array<T, N>& Array<T, N>::operator=(const std::initializer_list<T> &init_list)
{
    assert(init_list.size() == N);

    uint64_t idx = 0;

    for(T value : init_list)
    {
        this->data[idx] = value;
        idx++;
    }

    return *this;
}
