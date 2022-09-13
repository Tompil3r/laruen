
#ifndef LARUEN_NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_
#define LARUEN_NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/nditer.h"
#include "laruen/nn/losses/loss.h"
#include "laruen/nn/utils.h"

namespace laruen::nn::losses {

    using laruen::multi::NDArray;
    using laruen::multi::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class CategoricalCrossentropy : public Loss<T> {
            public:
                inline CategoricalCrossentropy() noexcept
                : Loss<T>("categorical crossentropy")
                {}

                inline CategoricalCrossentropy(const std::string &name) noexcept
                : Loss<T>(name)
                {}

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    assert(y_true.ndim() == 2 && y_pred.ndim() == 2 && y_true.size() == y_pred.size());

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += true_iter.next() * std::log(stable_nonzero(pred_iter.next())); 
                    }

                    return (-loss) / y_pred.shape().front();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &grad_output) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(grad_output.data(), grad_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        output_iter.next() = (-true_iter.next() / stable_nonzero(pred_iter.next()))
                        / batch_size;
                    }
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> categorical_crossentropy() noexcept {
            return std::shared_ptr<Loss<T>>(new CategoricalCrossentropy<T>());
        }

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> categorical_crossentropy(const std::string &name) {
            return std::shared_ptr<Loss<T>>(new CategoricalCrossentropy<T>(name));
        }
    }

    using namespace impl;
}


#endif