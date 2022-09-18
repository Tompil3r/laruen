
#ifndef LARUEN_NN_CALLBACKS_CALLBACK_H_
#define LARUEN_NN_CALLBACKS_CALLBACK_H_

#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"

namespace laruen::nn {

    namespace impl {
        template <typename> class Model;
    }

    namespace callbacks {

        namespace impl {

            using laruen::multi::NDArray;
            using laruen::multi::float32_t;
            using laruen::nn::Model;

            template <typename T = float32_t>
            class Callback {
                protected:
                    Model<T> *model_ = nullptr;
                    uint_fast8_t verbose_;

                public:
                    virtual ~Callback()
                    {}

                    Callback(uint_fast8_t verbose = 1)
                    : verbose_(verbose)
                    {}

                    inline void model(Model<T> *model) noexcept {
                        this->model_ = model;
                    }

                    inline Model<T>* model() noexcept {
                        return this->model_;
                    }

                    inline void verbose_mode(uint_fast8_t verbose) {
                        this->verbose_ = verbose;
                    }

                    inline uint_fast8_t verbose_mode() const {
                        return this->verbose_;
                    }

                    virtual Callback<T>* clone() const = 0;

                    virtual void on_epoch_end(uint_fast64_t epoch) const = 0;
            };
        }

        using namespace impl;
    }
}

#endif