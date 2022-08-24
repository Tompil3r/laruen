
#ifndef NN_LOSSES_SPARSE_CATEGORICAL_CROSSENTROPY_H_
#define NN_LOSSES_SPARSE_CATEGORICAL_CROSSENTROPY_H_

#include <cassert>
#include <cmath>
#include <limits>
#include "src/ndlib/ndarray.h"
#include "src/ndlib/nditer.h"
#include "src/ndlib/types.h"
#include "src/nn/losses/loss.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::ndlib::NDArray;
        using laruen::ndlib::NDIter;
        using laruen::ndlib::float32_t;

        template <typename T = float32_t>
        class SparseCategoricalCrossentropy : Loss<T> {
            public:
                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    assert(y_pred.ndim() == 2);
                    // y_true.shape = (batch_size, 1)
                    // y_pred.shape = (batch_size, classes)
                    
                    NDIter true_iter(y_true.data(), y_true);
                    uint_fast64_t pred_axis0 = 0; // axis 0 - batch axis
                    uint_fast64_t pred_axis0_stride = y_pred.strides().front();
                    uint_fast64_t pred_axis1_stride = y_pred.strides()[1]; // axis 1 - probabilities axis
                    T loss = 0;
                    T tmp;

                    for(uint_fast64_t i = 0;i < y_true.size();i++) {
                        loss += std::log((tmp = y_pred.data()[pred_axis0 + (uint_fast64_t)true_iter.next() * pred_axis1_stride])
                        ? tmp : std::numeric_limits<T>::min());
                        pred_axis0 += pred_axis0_stride;
                    }

                    return (-loss) / y_true.size();
                }

                void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    
                }
        };
    }

    using namespace impl;
}


#endif