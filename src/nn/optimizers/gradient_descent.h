
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
                T learning_rate_;

            public:
                GradientDescent(T learning_rate = 0.01f) noexcept
                : learning_rate_(learning_rate)
                {}

                void update(NDArray<T> &weights, NDArray<T> &raw_gradients, NDArray<T> &final_gradients) override final {
                    raw_gradients.multiply(this->learning_rate_, final_gradients);
                    weights.subtract_eq(final_gradients);
                }

                inline T learning_rate() const noexcept {
                    return this->learning_rate_;
                }

                inline void learning_rate(T learning_rate) noexcept {
                    this->learning_rate_ = learning_rate;
                }
        };
    }

    using namespace impl;
}

#endif