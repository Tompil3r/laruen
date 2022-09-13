
#ifndef LARUEN_NN_LOSSES_MEAN_SQUARED_ERROR_H_
#define LARUEN_NN_LOSSES_MEAN_SQUARED_ERROR_H_

#include <cassert>
#include <cmath>
#include <memory>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/nditer.h"
#include "laruen/multi/types.h"
#include "laruen/nn/losses/loss.h"

namespace laruen::nn::losses {

    using laruen::multi::NDArray;
    using laruen::multi::float32_t;
    using laruen::multi::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class MeanSquaredError : public Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    assert(y_true.shape() == y_pred.shape());

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;
                    T tmp;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        tmp = true_iter.next() - pred_iter.next();
                        loss += tmp * tmp;
                    }

                    return loss / y_pred.size();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &grad_output) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(grad_output.data(), grad_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        output_iter.next() = 2*(pred_iter.next() - true_iter.next()) / batch_size;
                    }
                }

                const char* name() const noexcept override final {
                    return "mean squared error";
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> mean_squared_error() noexcept {
            return std::shared_ptr<Loss<T>>(new MeanSquaredError<T>());
        }
    }

    using namespace impl;
}


#endif