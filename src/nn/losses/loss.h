
#ifndef NN_LOSSES_LOSE_H_
#define NN_LOSSES_LOSE_H_

#include "src/ndlib/ndarray.h"
#include "src/ndlib/types.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class Loss {

            public:
                virtual T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const = 0;
        };
    }

    using namespace impl;
}



#endif