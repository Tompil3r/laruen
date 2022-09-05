
#ifndef LARUEN_NN_OPTIMIZERS_RMSPROP_H_
#define LARUEN_NN_OPTIMIZERS_RMSPROP_H_

#include <memory>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/optimizers/optimizer.h"


namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::multi::NDArray;        
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class RMSprop : public Optimizer<T> {
            private:
                T rho_;
                T epsilon_;
            
            public:
                RMSprop(T learning_rate = 0.001f, T rho = 0.9f, T epsilon = 1e-7f)
                : Optimizer<T>(learning_rate), rho_(rho), epsilon_(epsilon)
                {}

                void update_weights(NDArray<T> &weights, NDArray<T> &raw_gradients,
                NDArray<T> &final_gradients, std::vector<NDArray<T>> &opt_caches) const override final
                {
                    // opt_caches.size = 1 (s_dw)
                    // s_dw initialized as 0's in first iteration
                    NDArray<T> &s_dw = opt_caches.front(); // = opt_caches[0]

                    s_dw.multiply_eq(this->rho_); // s_dw *= rho
                    
                    raw_gradients.multiply(raw_gradients, final_gradients); // using
                    // final_dw as computation memory: final_dw = dw*dw (element wise)

                    final_gradients.multiply_eq(1 - this->rho_); // final_dw *= (1 - rho)

                    s_dw.add_eq(final_gradients); // s_dw = rho * s_dw + (1 - rho)dw * dw

                    s_dw.power(0.5f, final_gradients); // final_dw = sqrt(s_dw)
                    final_gradients.add_eq(this->epsilon_); // final_dw += epsilon
                    final_gradients.inverse_divide_eq(raw_gradients); // final_dw = dw / final_dw

                    final_gradients.multiply_eq(this->learning_rate_); // final_dw *= lr

                    weights.subtract_eq(final_gradients); // w -= final_dw
                }

                inline void update_optimizer_params() noexcept override final
                {}

                inline constexpr uint_fast64_t required_caches() const noexcept override final {
                    return 1;
                }

                inline T rho() const noexcept {
                    return this->rho_;
                }

                inline void rho(T rho) {
                    this->rho_ = rho;
                }

                inline T epsilon() const noexcept {
                    return this->epsilon_;
                }

                inline void epsilon(T epsilon) {
                    this->epsilon_ = epsilon;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Optimizer<T>> shared_rmsprop(T learning_rate = 0.001f,
        T rho = 0.9f, T epsilon = 1e-7f) noexcept
        {
            return std::shared_ptr<Optimizer<T>>(new RMSprop<T>(learning_rate, rho, epsilon));
        }
    }

    using namespace impl;
}

#endif