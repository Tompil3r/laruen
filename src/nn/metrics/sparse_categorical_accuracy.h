
#ifndef NN_METRICS_SPARSE_CATEGORICAL_ACCURACY_H_
#define NN_METRICS_SPARSE_CATEGORICAL_ACCURACY_H_

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
        class SparseCategoricalAccuracy : public Metric<T> {            
            public:
                static constexpr char NAME[] = "sparse categorical accuracy";

                T operator()(const NDArray<T> &y_true, const NDArray<T> &y_pred) const override final {
                    NDIter true_iter(y_true.data(), y_true);
                    NDIter pred_iter(y_pred.data(), y_pred);

                    T score = 0;

                    for(uint_fast64_t i = 0;i < y_true.size();i++) {
                        T max = pred_iter.next();
                        uint_fast64_t index_max = 0;

                        for(uint_fast64_t j = 1;j < y_pred.shape().back();j++) {
                            if(pred_iter.current() > max) {
                                max = pred_iter.current();
                                index_max = j;
                            }
                            pred_iter.next();
                        }
                        score += true_iter.next() == index_max;
                    }

                    return (score / y_true.size());
                }

                const char* name() const noexcept override final {
                    return this->NAME;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Metric<T>> shared_sp_cat_acc() noexcept {
            return std::shared_ptr<Metric<T>>(new SparseCategoricalAccuracy<T>());
        }
    }

    using namespace impl;
}



#endif