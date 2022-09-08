
#ifndef LARUEN_NN_INITIALIZERS_ZEROS_H_
#define LARUEN_NN_INITIALIZERS_ZEROS_H_

#include <cstdint>
#include <cmath>
#include <memory>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/initializers/initializer.h"

namespace laruen::nn::initializers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Zeros : public Initializer<T> {
            public:
                inline void operator()(uint_fast64_t nb_inputs, uint_fast64_t nb_nodes,
                NDArray<T> &output) const override final
                {
                    output.fill((T)0);
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Initializer<T>> zeros() noexcept {
            return std::shared_ptr<Initializer<T>>(new Zeros<T>());
        }
    }

    using namespace impl;
}

#endif