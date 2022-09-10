
#ifndef LARUEN_NN_METRICS_METRIC_H_
#define LARUEN_NN_METRICS_METRIC_H_

#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"

namespace laruen::nn::metrics {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class Metric {
            public:
                virtual ~Metric()
                {}

                virtual T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const = 0;
                virtual const char* name() const noexcept = 0;
        };
    }

    using namespace impl;
}

#endif