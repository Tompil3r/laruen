
#ifndef LARUEN_NN_LOSSES_LOSE_H_
#define LARUEN_NN_LOSSES_LOSE_H_

#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/metrics/metric.h"

namespace laruen::nn::losses {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::metrics::Metric;

        template <typename T = float32_t>
        class Loss : public Metric<T> {
            public:
                virtual ~Loss()
                {}

                virtual T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const = 0;
                virtual void backward(const NDArray<T> &y_true, const NDArray<T> &y_pred, NDArray<T> &grad_output) const = 0;
                virtual const char* name() const noexcept = 0;
        };
    }

    using namespace impl;
}



#endif