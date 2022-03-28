
#include "laruen/utils/container.h"
#include <cstdint>
#include <initializer_list>
#include <cassert>

using namespace laruen::utils;

template <typename T, typename U, U N>
Container<T, U, N>::~Container() {
    delete[] this->data;
}

template <typename T, typename U, U N>
Container<T, U, N>::Container() : data(new T[N]) {}

template <typename T, typename U, U N>
Container<T, U, N>::Container(T value) : data(new T[N]) {
    for(U idx = 0;idx < N;idx++) {
        this->data[idx] = value;
    }
}

template <typename T, typename U, U N>
Container<T, U, N>::Container(const std::initializer_list<T> &init_list) : data(new T[N]) {
    assert(init_list.size() == N);

    U idx = 0;

    for(T value : init_list) {
        this->data[idx] = value;
        idx++;
    }
}

template <typename T, typename U, U N>
Container<T, U, N>::Container(const Container<T, U, N> &container) : data(new T[N]) {
    for(U idx = 0;idx < N;idx++) {
        this->data[idx] = container[idx];
    }
}

template <typename T, typename U, U N>
Container<T, U, N>::Container(Container<T, U, N> &&container) : data(container.data) {
    container.data = nullptr;
}

template <typename T, typename U, U N>
Container<T, U, N>& Container<T, U, N>::operator=(const Container<T, U, N> &container) {
    for(U idx = 0;idx < N;idx++) {
        this->data[idx] = container[idx];
    }

    return *this;
}

template <typename T, typename U, U N>
Container<T, U, N>& Container<T, U, N>::operator=(Container<T, U, N> &&container) {
    if(this == &container) {return this;}

    delete[] this->data;
    this->data = container.data;
    container.data = nullptr;

    return *this;
}

template <typename T, typename U, U N>
Container<T, U, N>& Container<T, U, N>::operator=(const std::initializer_list<T> &init_list) {
    assert(init_list.size() == N);

    U idx = 0;

    for(T value : init_list) {
        this->data[idx] = value;
        idx++;
    }

    return *this;
}