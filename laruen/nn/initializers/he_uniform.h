
#ifndef LARUEN_NN_INITIALIZERS_HE_UNIFORM_H_
#define LARUEN_NN_INITIALIZERS_HE_UNIFORM_H_

#include <cstdint>
#include <cmath>
#include <memory>
#include <random>
#include "laruen/multi/ndarray.h"
#include "laruen/multi/types.h"
#include "laruen/multi/rng.h"
#include "laruen/nn/initializers/initializer.h"

namespace laruen::nn::initializers {

    namespace impl {
        using laruen::multi::NDArray;
        using laruen::multi::float32_t;

        template <typename T = float32_t>
        class HeUniform : public Initializer<T> {
            private:
                std::mt19937::result_type seed_;
                uint_fast64_t count_ = 0;

            public:
                inline HeUniform(std::mt19937::result_type seed = std::random_device{}(),
                uint_fast64_t count = 0) noexcept
                : seed_(seed), count_(count)
                {}

                inline void operator()(uint_fast64_t nb_inputs, uint_fast64_t nb_nodes,
                NDArray<T> &output) override final
                {
                    laruen::multi::RNG.seed(this->seed_);
                    laruen::multi::RNG.discard(this->count_);

                    T limit = std::sqrt((T)6 / nb_inputs);
                    output.rand(-limit, limit);

                    this->count_ += output.size();
                }

                inline Initializer<T>* clone() const override final {
                    return new HeUniform<T>(*this);
                }

                inline void seed(std::mt19937::result_type seed) noexcept {
                    this->seed_ = seed;
                }
                
                inline std::mt19937::result_type seed() const noexcept {
                    return this->seed_;
                }

                inline void count(uint_fast64_t count) noexcept {
                    this->count_ = count;
                }

                inline uint_fast64_t count() noexcept {
                    return this->count_;
                }
        };

        template <typename T = float32_t>
        inline std::shared_ptr<Initializer<T>> he_uniform(
            std::mt19937::result_type seed = std::random_device{}(), uint_fast64_t count = 0) noexcept
        {
            return std::shared_ptr<Initializer<T>>(new HeUniform<T>(seed, count));
        }
    }

    using namespace impl;
}

#endif