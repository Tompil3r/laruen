
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
            public:
                virtual void update(NDArray<T> &weights, NDArray<T> &raw_gradients, NDArray<T> &final_gradients) = 0;
        };

    }

    using namespace impl;
}


#endif
