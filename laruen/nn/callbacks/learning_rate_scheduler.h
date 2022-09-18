
#ifndef LARUEN_NN_CALLBACKS_LEARNING_RATE_SCHEDULER_H_
#define LARUEN_NN_CALLBACKS_LEARNING_RATE_SCHEDULER_H_

#include <cstdint>
#include <memory>
#include <string>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"
#include "laruen/nn/callbacks/callback.h"

namespace laruen::nn::callbacks {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::Model;

        template <typename T = float32_t>
        class LearningRateScheduler : public Callback<T> {
            private:
                T (*scheduler_)(uint_fast64_t, T);

            public:
                LearningRateScheduler(T (*scheduler)(uint_fast64_t, T), uint_fast8_t verbose = 1)
                : Callback<T>(verbose), scheduler_(scheduler)
                {}

                Callback<T>* clone() const override final {
                    return new LearningRateScheduler(this->scheduler_, this->verbose_mode_);
                }

                void on_epoch_end(uint_fast64_t epoch) override final {
                    this->model_->optimizer()->learning_rate(
                        this->scheduler_(epoch, this->model_->optimizer()->learning_rate()));
                        
                    if(this->verbose_mode_) {
                        this->verbose_.clear();
                        std::string lr_str = std::to_string(this->model_->optimizer()->learning_rate());

                        this->verbose_.append("learning_rate: ");
                        this->verbose_.append(lr_str.cbegin(), lr_str.cbegin() + lr_str.find_last_not_of('0') + 1);
                    }
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Callback<T>> learning_rate_scheduler(T (*scheduler)(uint_fast64_t, T),
        uint_fast8_t verbose = 1)
        {
            return std::shared_ptr<Callback<T>>(new LearningRateScheduler<T>(scheduler, verbose));
        }
    }

    using namespace impl;
}

#endif