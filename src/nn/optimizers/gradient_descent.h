
#ifndef NN_OPTIMIZERS_GRADIENT_DESCENT_H_
#define NN_OPTIMIZERS_GRADIENT_DESCENT_H_

#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/optimizers/optimizer.h"

namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class GradientDescent : public Optimizer<T> {
            private:
                T momentum_;

            public:
                GradientDescent(T learning_rate = 0.01f, T momentum = 0.0) noexcept
                : Optimizer<T>(learning_rate), momentum_(momentum)
                {}

                void update(NDArray<T> &weights, NDArray<T> &raw_gradients,
                NDArray<T> &final_gradients, std::vector<NDArray<T>> &opt_caches) override final
                {
                    raw_gradients.multiply(this->learning_rate_, final_gradients);
                    weights.subtract_eq(final_gradients);
                }

                inline constexpr uint_fast64_t required_caches() const noexcept override final {
                    return 1;
                }
        };
    }

    using namespace impl;
}

#endif