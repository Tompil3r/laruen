
#ifndef NN_LOSSES_MEAN_ABSOLUTE_ERROR_H_
#define NN_LOSSES_MEAN_ABSOLUTE_ERROR_H_

#include <cassert>
#include <cmath>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/types.h"
#include "src/nn/losses/loss.h"

namespace laruen::nn::losses {

    using laruen::ndlib::NDArray;
    using laruen::ndlib::float32_t;
    using laruen::ndlib::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class MeanAbsoluteError : Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    assert(y_true.shape() == y_pred.shape());

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += std::abs(true_iter.next() - pred_iter.next());
                    }

                    return loss / y_pred.size();
                }
        };
    }

    using namespace impl;
}


#endif