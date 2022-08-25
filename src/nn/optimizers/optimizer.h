
#ifndef NN_OPTIMIZERS_OPTIMIZER_H_
#define NN_OPTIMIZERS_OPTIMIZER_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"

namespace laruen::nn::optimizers {

    namespace impl {
        using laruen::ndlib::NDArray;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class Optimizer {
            public:
                virtual void update(NDArray<T> &weights, const NDArray<T> &gradients) = 0;
        };

    }

    using namespace impl;
}


#endif
