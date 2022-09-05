
#ifndef LARUEN_NN_LOSSES_LOSE_H_
#define LARUEN_NN_LOSSES_LOSE_H_

#include "src/multi/ndarray.h"
#include "src/multi/types.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Loss {
            public:
                virtual ~Loss()
                {}

                virtual T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const = 0;
                virtual void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &deriv_output) const = 0;
        };
    }

    using namespace impl;
}



#endif