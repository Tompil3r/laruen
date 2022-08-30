
#ifndef NN_OPTIMIZERS_ADAM_H_
#define NN_OPTIMIZERS_ADAM_H_

#include <cmath>
#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/nn/optimizers/optimizer.h"

namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Adam : public Optimizer<T> {
            private:
                T beta1_;
                T beta2_;
                T epsilon_;
                T beta1_correction_;
                T beta2_correction_;
            
            public:
                Adam(T learning_rate = 0.001f, T beta1 = 0.9f, T beta2 = 0.999f, T epsilon = 1e-7f)
                : Optimizer<T>(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
                beta1_correction_(beta1), beta2_correction_(beta2)
                {}

                void update_weights(NDArray<T> &weights, NDArray<T> &raw_gradients,
                NDArray<T> &final_gradients, std::vector<NDArray<T>> &opt_caches) override final
                {
                    // opt_caches.size = 3
                    // opt_caches = v_dw, s_dw, tmp (momentum, rmsprop, memory for computation = tmp)
                    NDArray<T> &v_dw = opt_caches.front(); // = opt_caches[0]
                    NDArray<T> &s_dw = opt_caches[1];
                    NDArray<T> &tmp = opt_caches.back(); // = opt_caches[2]

                    // --- v_dw computation ---
                    v_dw.multiply_eq(this->beta1_); // v_dw *= beta1
                    
                    // using final_dw as computation memory
                    raw_gradients.multiply(1 - this->beta1_, final_gradients);

                    v_dw.add_eq(final_gradients); // v_dw = beta1 * v_dw + (1 - beta1) * dw

                    // v_dw bias correction (stored in final_dw) (t = current iteration number)
                    v_dw.divide(1 - this->beta1_correction_, final_gradients); // final_dw = v_dw / (1 - beta1**t)

                    // --- s_dw computation ---
                    s_dw.multiply_eq(this->beta2_); // s_dw *= beta2

                    // using final_dw as computation memory
                    raw_gradients.multiply(raw_gradients, tmp); // tmp = dw * dw
                    
                    tmp.multiply_eq(1 - this->beta2_); // tmp = (1 - beta2) * dw * dw

                    s_dw.add_eq(tmp); // s_dw = beta2 * s_dw + (1 - beta2) * dw * dw 

                    // s_dw bias correction (stored in tmp) (t = current iteration number)
                    s_dw.divide(1 - this->beta2_correction_, tmp); // tmp = s_dw / (1 - beta2**t)

                    tmp.power_eq(0.5); // tmp := sqrt(tmp) = sqrt(s_dw_corrected)

                    tmp.add_eq(this->epsilon_); // tmp := tmp + epsilon = sqrt(s_dw_corrected) + epsilon =
                    // the denominator of the update component: (w -= lr * v_dw_cor / (sqrt(s_dw_cor) + epsilon))

                    // variable reminder: final_gradients = v_dw_corrected (numerator),
                    // tmp = sqrt(s_dw_corrected) + epsilon (denominator)

                    final_gradients.divide_eq(tmp); // final_gradients := numerator
                    // component / denominator component

                    // multiply by learning_rate -> final gradients calculation
                    final_gradients.multiply_eq(this->learning_rate_); // final_dw *= lr

                    // update weights
                    weights.subtract_eq(final_gradients); // weights -= final_dw

                    // update bias correction for beta1 and beta2
                    this->beta1_correction_ *= this->beta1_;
                    this->beta2_correction_ *= this->beta2_;
                }

                inline constexpr uint_fast64_t required_caches() const noexcept override final {
                    return 3;
                }

                inline T beta1() const noexcept {
                    return this->beta1_;
                }

                inline void beta1(T beta1) {
                    this->beta1_ = beta1;
                }

                inline T beta2() const noexcept {
                    return this->beta2_;
                }

                inline void beta2(T beta2) {
                    this->beta2_ = beta2;
                }

                inline T epsilon() const noexcept {
                    return this->epsilon_;
                }

                inline void epsilon(T epsilon) {
                    this->epsilon_ = epsilon;
                }

                inline T beta1_correction() const noexcept {
                    return this->beta1_correction_;
                }

                inline T beta2_correction() const noexcept {
                    return this->beta2_correction_;
                }

                inline void reset_iteration(uint_fast64_t iteration_nb = 1) {
                    this->beta1_correction_ = std::pow(this->beta1_, iteration_nb);
                    this->beta2_correction_ = std::pow(this->beta2_, iteration_nb);
                }
        };
    }

    using namespace impl;
}

#endif