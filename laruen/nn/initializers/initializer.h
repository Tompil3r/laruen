
#ifndef LARUEN_NN_INITIALIZERS_INITIALIZER_H_
#define LARUEN_NN_INITIALIZERS_INITIALIZER_H_

#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"

namespace laruen::nn::initializers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Initializer {
            public:
                virtual Initializer<T>* clone() const = 0;

                virtual void operator()(uint_fast64_t nb_inputs, uint_fast64_t nb_nodes, NDArray<T> &output) const = 0;
        };
    }

    using namespace impl;
}


#endif