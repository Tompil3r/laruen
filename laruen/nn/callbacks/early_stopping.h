
#ifndef LARUEN_NN_CALLBACKS_EARLY_STOPPING_H_
#define LARUEN_NN_CALLBACKS_EARLY_STOPPING_H_

#include <cstdint>
#include <memory>
#include <string>
#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"
#include "laruen/nn/callbacks/callback.h"
#include "laruen/nn/metrics/metric.h"

namespace laruen::nn::callbacks {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::Model;
        using laruen::nn::metrics::Metric;

        template <typename T = float32_t>
        class EarlyStopping : public Callback<T> {
            private:
                const Metric<T> *monitor_;
                T mode_;
                T best_monitored_;
                T min_delta_;
                uint_fast64_t patience_;
                uint_fast64_t patience_count_ = 0;

            public:
                EarlyStopping(const Metric<T> &monitor, T mode, T min_delta, uint_fast64_t patience,
                uint_fast8_t verbose)
                : Callback<T>(verbose), monitor_(&monitor), mode_(mode), min_delta_(min_delta), patience_(patience)
                {}

                inline EarlyStopping(const Metric<T> &monitor, T min_delta = 0.0, uint_fast64_t patience = 0,
                uint_fast8_t verbose = 1)
                : EarlyStopping<T>(monitor, monitor->optimizing_mode(), min_delta, patience, verbose)
                {}

                inline Callback<T>* clone() const override final {
                    return new EarlyStopping(this->monitor_, this->mode_,
                    this->min_delta_, this->patience_, this->verbose_);
                }

                inline void set(Model<T> *model) override final {
                    this->model_ = model;
                    this->patience_count_ = 0;
                }

                bool on_epoch_end(uint_fast64_t epoch) override final {
                    // on first epoch
                    if(!epoch) {
                        this->best_monitored_ = this->monitor_->values().front();
                        return false;
                    }

                    T curr_metric_value = this->monitor_->values()[epoch];

                    if((curr_metric_value - this->best_monitored_) * this->mode_ < this->min_delta_) {
                        this->patience_count_++;
                    }

                    return this->patience_count_ > this->patience_;
                }

                inline void monitor(const Metric<T> *monitor) noexcept {
                    this->monitor_ = monitor;
                }

                inline const Metric<T>* monitor() noexcept {
                    return this->monitor_;
                }

                inline void mode(T mode) {
                    this->mode_ = mode;
                }

                inline T mode() const noexcept {
                    return this->mode_;
                }

                inline void best_monitored(T best_monitored) noexcept {
                    this->best_monitored_ = best_monitored;
                }

                inline T best_monitored() const noexcept {
                    return this->best_monitored_;
                }

                inline void min_delta(T min_delta) noexcept {
                    this->min_delta_ = min_delta;
                }

                inline T min_delta() const noexcept {
                    return this->min_delta_;
                }

                inline void patience(uint_fast64_t patience) noexcept {
                    this->patience_ = patience;
                }

                inline uint_fast64_t patience() const noexcept {
                    return this->patience_;
                }

                inline void patience_count(uint_fast64_t patience_count) noexcept {
                    this->patience_count_ = patience_count;
                }
                
                inline uint_fast64_t patience_count() const noexcept {
                    return this->patience_count_;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Callback<T>> early_stopping(const Metric<T> &monitor,
        T mode, T min_delta, uint_fast64_t patience, uint_fast8_t verbose)
        {
            return std::shared_ptr<Callback<T>>(new EarlyStopping<T>(monitor, mode,
            min_delta, patience, verbose));
        }

        template <typename T = float32_t>
        inline std::shared_ptr<Callback<T>> early_stopping(const Metric<T> &monitor,
        T min_delta = 0.0, uint_fast64_t patience = 0, uint_fast8_t verbose = 1)
        {
            return std::shared_ptr<Callback<T>>(new EarlyStopping<T>(monitor,
            min_delta, patience, verbose));
        }
    }

    using namespace impl;
}

#endif