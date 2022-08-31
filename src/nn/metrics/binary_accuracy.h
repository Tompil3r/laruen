
#ifndef NN_METRICS_BINARY_ACCURACY_H_
#define NN_METRICS_BINARY_ACCURACY_H_

#include "src/multi/ndarray.h"
#include "src/multi/types.h"
#include "src/multi/nditer.h"
#include "src/nn/metrics/metric.h"

namespace laruen::nn::metrics {

    using laruen::multi::NDArray;
    using laruen::multi::float32_t;
    using laruen::multi::NDIter;

    namespace impl {
        
        template <typename T = float32_t>
        class BinaryAccuracy : public Metric<T> {
            private:
                T threshold_;
            
            public:
                static constexpr char NAME[] = "binary accuracy";

                BinaryAccuracy(T threshold = 0.5f)
                : threshold_(threshold)
                {}

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);

                    T score = 0;

                    for(uint_fast64_t i = 0;i < y_pred.size();i++) {
                        score += pred_iter.next() >= this->threshold_ ?
                        ((T)1.0f) == true_iter.next() : ((T)0.0f) == true_iter.next();
                    }

                    return (score / y_pred.size());
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }
        };
    }

    using namespace impl;
}



#endif