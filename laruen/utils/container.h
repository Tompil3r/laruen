
#ifndef CONTAINER_H
#define CONTAINER_H

#include <cstdint>
#include <initializer_list>

namespace laruen::utils
{
    template <typename T, typename U, U N> class Container
    {
        T *data;

        public:
            ~Container();
            Container();
            Container(T value);
            Container(const std::initializer_list<T> &init_list);
            Container(const Container &container);
            Container(Container &&container);
            Container& operator=(const Container &container);
            Container& operator=(Container &&container);
            Container& operator=(const std::initializer_list<T> &init_list);

            inline T& operator[](U index)
            {
                return this->data[index];
            }

            inline const T& operator[](U index) const
            {
                return this->data[index];
            }

            inline constexpr U size() const
            {
                return N;
            }
    };
};

#include "laruen/utils/container.tpp"
#endif