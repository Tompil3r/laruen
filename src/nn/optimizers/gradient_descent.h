
#ifndef NN_OPTIMIZERS_GRADIENT_DESCENT_H_
#define NN_OPTIMIZERS_GRADIENT_DESCENT_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"
#include "src/nn/optimizers/optimizer.h"

namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::ndlib::NDArray;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class GradientDescent : public Optimizer<T> {
            private:
                T learning_rate_;

            public:
                GradientDescent(T learning_rate = 0.01f) noexcept
                : learning_rate_(learning_rate)
                {}

                void update(NDArray<T> &weights, NDArray<T> &gradients) override final {
                    gradients.multiply_eq(this->learning_rate_);
                    weights.subtract_eq(gradients);
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