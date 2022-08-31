
#ifndef NN_LOSSES_MEAN_ABSOLUTE_ERROR_H_
#define NN_LOSSES_MEAN_ABSOLUTE_ERROR_H_

#include <cassert>
#include <cmath>
#include <memory>
#include "src/multi/ndarray.h"
#include "src/multi/nditer.h"
#include "src/multi/types.h"
#include "src/nn/losses/loss.h"
#include "src/math/common.h"
#include "src/math/utils.h"

namespace laruen::nn::losses {

    using laruen::multi::NDArray;
    using laruen::multi::float32_t;
    using laruen::multi::NDIter;

    namespace impl {

        template <typename T = float32_t>
        class MeanAbsoluteError : public Loss<T> {
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

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &deriv_output) const override final {
                    using laruen::math::common::sign;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(deriv_output.data(), deriv_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        // dL =
                        // -1 if y_pred < y_true
                        // 0 if y_pred == y_true
                        // 1 if y_pred > y_true
                        output_iter.next() = sign(pred_iter.next() - true_iter.next()) / batch_size;
                    }
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> shared_mae() noexcept {
            return std::shared_ptr<Loss<T>>(new MeanAbsoluteError<T>());
        }
    }

    using namespace impl;
}


#endif