
#ifndef LARUEN_NN_CALLBACKS_CALLBACK_H_
#define LARUEN_NN_CALLBACKS_CALLBACK_H_

#include <cstdint>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/nn/model.h"

namespace laruen::nn::callbacks {

    namespace impl {

        using laruen::multi::NDArray;
        using laruen::multi::float32_t;
        using laruen::nn::Model;

        template <typename T = float32_t>
        class Callback {
            private:
                Model<T> *model_ = nullptr;
                uint_fast8_t verbose_;

            public:
                virtual ~Callback()
                {}

                Callback(uint_fast8_t verbose = 1)
                : verbose_(verbose)
                {}

                virtual Callback<T>* clone() const = 0;

                virtual void operator()() const = 0;
        };
    }

    using namespace impl;
}

#endif