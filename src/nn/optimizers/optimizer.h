
#ifndef NN_OPTIMIZERS_OPTIMIZER_H_
#define NN_OPTIMIZERS_OPTIMIZER_H_

#include "src/multi/ndarray.h"
#include "src/multi/types.h"

namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Optimizer {
            protected:
                T learning_rate_;

            public:
                Optimizer(T learning_rate) noexcept
                : learning_rate_(learning_rate)
                {}

                virtual void update(NDArray<T> &weights, NDArray<T> &raw_gradients, NDArray<T> &final_gradients) = 0;

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
