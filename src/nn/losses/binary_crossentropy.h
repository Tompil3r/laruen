
#ifndef NN_LOSSES_BINARY_CROSSENTROPY_H_
#define NN_LOSSES_BINARY_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/types.h"
#include "src/nn/losses/loss.h"
#include "src/math/utils.h"

namespace laruen::nn::losses {

    using laruen::ndlib::NDArray;
    using laruen::ndlib::float32_t;
    using laruen::ndlib::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class BinaryCrossentropy : Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    using laruen::math::utils::nonzero;

                    assert(y_true.ndim() == 2 && y_pred.ndim() == 2 && y_true.size() == y_pred.size());
                    // y_true.shape = y_pred.shape = (batch size, 1)

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;
                    T tmp;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += true_iter.current() * std::log(nonzero(pred_iter.current()))
                        + (1 - true_iter.current()) * std::log(nonzero(1 - pred_iter.current()));

                        true_iter.next();
                        pred_iter.next();
                    }

                    return (-loss) / y_pred.size();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &deriv_output) const override final {
                    using math::utils::nonzero;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(deriv_output.data(), deriv_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        // (dL / dy_hat[i]) = ((1 - y[i]) / (1 - y_hat[i]) - y[i] / y_hat[i]) / batch_size
                        output_iter.next() = ((1 - true_iter.current()) / nonzero(1 - pred_iter.current())
                        - true_iter.current() / nonzero(pred_iter.current())) / batch_size;

                        true_iter.next();
                        pred_iter.next();
                    }
                }
        };
    }

    using namespace impl;
}


#endif