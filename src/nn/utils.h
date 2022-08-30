
#ifndef NN_UTILS_H_
#define NN_UTILS_H_

#include <cstdint>
#include <algorithm>
#include "src/multi/types.h"

namespace laruen::nn::utils {

    namespace impl {

        using laruen::multi::Shape;

        Shape add_batch_shape(const Shape &shape, uint_fast64_t batch_size) noexcept {
            Shape result(shape.size() + 1);
            std::copy(shape.cbegin(), shape.cend(), result.begin() + 1);
            result.front() = batch_size;

            return result;
        }
    }

    using namespace impl;
}

#endif