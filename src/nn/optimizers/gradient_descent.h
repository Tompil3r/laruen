
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
                T momentum_; // sometimes referred to as beta

            public:
                GradientDescent(T learning_rate = 0.01f, T momentum = 0.0) noexcept
                : Optimizer<T>(learning_rate), momentum_(momentum)
                {}

                void update(NDArray<T> &weights, NDArray<T> &raw_gradients,
                NDArray<T> &final_gradients, std::vector<NDArray<T>> &opt_caches) override final
                {
                    // opt_caches relevant only when momentum > 0

                    if(this->momentum_ > 0) {
                        // opt_caches.size = 1 (v_dw = velocity)
                        // v_dw initialized as 0's in first iteration
                        NDArray<T> &v_dw = opt_caches.front(); // = opt_caches[0]

                        v_dw.multiply_eq(this->momentum_); // v_dw *= momentum (beta)
                        raw_gradients.multiply(1 - this->momentum_, final_gradients); // uses final_dw as calculation memory
                        v_dw.add_eq(final_gradients); // v_dw += (1 - momentum) * dw
                        // v_dw = momentum * v_dw + (1 - momentum) * dw

                        v_dw.multiply(this->learning_rate_, final_gradients); // final_dw = lr * v_dw
                    }

                    else {
                        raw_gradients.multiply(this->learning_rate_, final_gradients); // final_dw = lr * dw
                    }

                    weights.subtract_eq(final_gradients); // w -= final_dw
                }

                inline constexpr uint_fast64_t required_caches() const noexcept override final {
                    return 1;
                }

                inline T momentum() const noexcept {
                    return this->momentum_;
                }

                inline void momentum(T momentum) noexcept {
                    this->momentum_ = momentum;
                }
        };
    }

    using namespace impl;
}

#endif