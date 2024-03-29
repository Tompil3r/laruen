
#ifndef LARUEN_NN_LOSSES_BINARY_CROSSENTROPY_H_
#define LARUEN_NN_LOSSES_BINARY_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/nditer.h"
#include "laruen/multi/types.h"
#include "laruen/nn/losses/loss.h"
#include "laruen/nn/utils/utils.h"

namespace laruen::nn::losses {

    using laruen::multi::NDArray;
    using laruen::multi::float32_t;
    using laruen::multi::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class BinaryCrossentropy : public Loss<T> {
            public:
                inline BinaryCrossentropy() noexcept
                : Loss<T>("binary_crossentropy")
                {}

                inline BinaryCrossentropy(const std::string &name)
                : Loss<T>(name)
                {}

                inline Loss<T>* clone() const override final {
                    return new BinaryCrossentropy<T>(*this);
                }

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    assert(y_true.ndim() == 2 && y_pred.ndim() == 2 && y_true.size() == y_pred.size());
                    // y_true.shape = y_pred.shape = (batch size, 1)

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += true_iter.current() * std::log(stable_nonzero(pred_iter.current()))
                        + (1 - true_iter.current()) * std::log(stable_nonzero(1 - pred_iter.current()));

                        true_iter.next();
                        pred_iter.next();
                    }

                    return (-loss) / y_pred.size();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &grad_output) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(grad_output.data(), grad_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        // (dL / dy_hat[i]) = ((1 - y[i]) / (1 - y_hat[i]) - y[i] / y_hat[i]) / batch_size
                        output_iter.next() = ((1 - true_iter.current()) / stable_nonzero(1 - pred_iter.current())
                        - true_iter.current() / stable_nonzero(pred_iter.current())) / batch_size;

                        true_iter.next();
                        pred_iter.next();
                    }
                }

                inline T optimizing_mode() const override final {
                    return -1.0;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> binary_crossentropy() noexcept {
            return std::shared_ptr<Loss<T>>(new BinaryCrossentropy<T>());
        }

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> binary_crossentropy(const std::string &name) {
            return std::shared_ptr<Loss<T>>(new BinaryCrossentropy<T>(name));
        }
    }

    using namespace impl;
}


#endif