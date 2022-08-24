
#ifndef NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_
#define NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/nditer.h"
#include "src/nn/losses/loss.h"

namespace laruen::nn::losses {

    using laruen::ndlib::NDArray;
    using laruen::ndlib::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class CategoricalCrossentropy : Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    assert(y_true.ndim() == 2 && y_pred.ndim() == 2 && y_true.size() == y_pred.size());

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;
                    T tmp;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += true_iter.next() * std::log((tmp = pred_iter.next()) ? tmp : std::numeric_limits<T>::min()); 
                    }

                    return (-loss) / y_pred.shape().front();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    
                }
        };

    }

    using namespace impl;
}


#endif