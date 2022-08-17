
#ifndef NN_LOSSES_LOSE_H_
#define NN_LOSSES_LOSE_H_

#include "src/ndlib/ndarray.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::ndlib::NDArray;

        template <typename T>
        class Loss {

            public:
                T operator()(const NDArray<T> &y_pred, const NDArray<T> &y_true) const = 0;
        };
    }

    using namespace impl;
}



#endif