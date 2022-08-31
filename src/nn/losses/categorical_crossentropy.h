
#ifndef NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_
#define NN_LOSSES_CATEGORICAL_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include "src/multi/ndarray.h"
#include "src/multi/nditer.h"
#include "src/nn/losses/loss.h"
#include "src/math/utils.h"

namespace laruen::nn::losses {

    using laruen::multi::NDArray;
    using laruen::multi::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class CategoricalCrossentropy : public Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    using laruen::math::utils::nonzero;

                    assert(y_true.ndim() == 2 && y_pred.ndim() == 2 && y_true.size() == y_pred.size());

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    T loss = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        loss += true_iter.next() * std::log(nonzero(pred_iter.next())); 
                    }

                    return (-loss) / y_pred.shape().front();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &deriv_output) const override final {
                    using laruen::math::utils::nonzero;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(deriv_output.data(), deriv_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        output_iter.next() = (-true_iter.next() / nonzero(pred_iter.next()))
                        / batch_size;
                    }
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> shared_cce() noexcept {
            return std::shared_ptr<Loss<T>>(new CategoricalCrossentropy<T>());
        }
    }

    using namespace impl;
}


#endif