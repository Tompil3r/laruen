
#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <initializer_list>

namespace laruen::utils
{
    template <typename T, uint64_t N> class Array
    {
        T *data;

        public:
            ~Array();
            Array();
            Array(T value);
            Array(const std::initializer_list<T> &init_list);
            Array(const Array &array);
            Array& operator=(const std::initializer_list<T> &init_list);

            inline T& operator[](uint64_t index)
            {
                return this->data[index];
            }

            inline const T& operator[](uint64_t index) const
            {
                return this->data[index];
            }

            inline constexpr uint64_t size() const
            {
                return N;
            }
    };
};

#include "laruen/utils/array.tpp"
#endif