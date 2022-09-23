
#include <iostream>
#include "laruen/multi/ndarray.h"
#include "laruen/nn/model.h"
#include "laruen/nn/layers/fully_connected.h"
#include "laruen/nn/layers/sigmoid.h"
#include "laruen/nn/layers/relu.h"
#include "laruen/nn/losses/binary_crossentropy.h"
#include "laruen/nn/optimizers/adam.h"
#include "laruen/nn/metrics/binary_accuracy.h"

int main() {
    using namespace laruen;

    multi::NDArray<> x({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, multi::Shape{4, 2});
    multi::NDArray<> y({0.0, 1.0, 1.0, 0.0}, multi::Shape{4, 1});

    nn::Model model({
        nn::layers::fully_connected(4),
        nn::layers::relu(),
        nn::layers::fully_connected(1),
        nn::layers::sigmoid()
    });

    model.build({2});

    model.summary();

    model.compile(nn::optimizers::adam(),
    nn::losses::binary_crossentropy("loss"),
    {nn::metrics::binary_accuracy("accuracy")});

    auto train_history = model.fit(x, y, {}, 32, 1000);

    auto eval_history = model.evaluate(x, y);

    std::cout << model.predict(x, 32, 0) << std::endl;

    model.save_weights("examples/xor/xor_weights");

    return 0;
}