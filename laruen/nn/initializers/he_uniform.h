
#ifndef LARUEN_NN_INITIALIZERS_HE_UNIFORM_H_
#define LARUEN_NN_INITIALIZERS_HE_UNIFORM_H_

#include <cstdint>
#include <cmath>
#include <memory>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/initializers/initializer.h"

namespace laruen::nn::initializers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class HeUniform : public Initializer<T> {
            public:
                inline void operator()(uint_fast64_t nb_inputs, uint_fast64_t nb_nodes,
                NDArray<T> &output) const override final
                {
                    T limit = std::sqrt((T)6 / nb_inputs);
                    output.rand(-limit, limit);
                }

                inline Initializer<T>* clone() const override final {
                    return new HeUniform<T>(*this);
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Initializer<T>> he_uniform() noexcept {
            return std::shared_ptr<Initializer<T>>(new HeUniform<T>());
        }
    }

    using namespace impl;
}

#endif