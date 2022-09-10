
#ifndef LARUEN_NN_LOSSES_SPARSE_CATEGORICAL_CROSSENTROPY_H_
#define LARUEN_NN_LOSSES_SPARSE_CATEGORICAL_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/nditer.h"
#include "laruen/multi/types.h"
#include "laruen/nn/losses/loss.h"
#include "laruen/nn/utils.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::NDIter;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class SparseCategoricalCrossentropy : public Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    assert(y_pred.ndim() == 2);
                    // y_true.shape = (batch_size, 1)
                    // y_pred.shape = (batch_size, classes)
                    
                    NDIter true_iter(y_true.data(), y_true);
                    uint_fast64_t pred_axis0 = 0; // axis 0 - batch axis
                    uint_fast64_t pred_axis0_stride = y_pred.strides().front();
                    uint_fast64_t pred_axis1_stride = y_pred.strides()[1]; // axis 1 - probabilities axis
                    T loss = 0;

                    for(uint_fast64_t i = 0;i < y_true.size();i++) {
                        loss += std::log(stable_nonzero(y_pred.data()[pred_axis0 + (uint_fast64_t)true_iter.next() * pred_axis1_stride]));
                        pred_axis0 += pred_axis0_stride;
                    }

                    return (-loss) / y_true.size();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &deriv_output) const override final {
                    using laruen::nn::utils::stable_nonzero;

                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);
                    NDIter output_iter(deriv_output.data(), deriv_output);

                    uint_fast64_t batch_size = y_pred.shape().front();

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        if(pred_iter.ndindex.back() == true_iter.current()) {
                            output_iter.next() = ((T)(-1)/stable_nonzero(pred_iter.current())) / batch_size;
                        }
                        else {
                            output_iter.next() = 0;
                        }

                        pred_iter.next();

                        if(!((i + 1) % y_pred.shape().back())) {
                            true_iter.next();
                        }
                    }
                }

                const char* name() const noexcept override final {
                    return "sparse categorical crossentropy";
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Loss<T>> sparse_categorical_crossentropy() noexcept {
            return std::shared_ptr<Loss<T>>(new SparseCategoricalCrossentropy<T>());
        }
    }

    using namespace impl;
}


#endif