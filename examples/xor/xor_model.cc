
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

    // data: all 4 combinations of xor operation between 2 bits
    multi::NDArray<> x({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, multi::Shape{4, 2});
    multi::NDArray<> y({0.0, 1.0, 1.0, 0.0}, multi::Shape{4, 1});

    // the model to train on the xor operation, contains 4 layers.
    // different layer configurations can be tested to get different
    // results, for example: number of nodes in each 'fully_connected'
    // layer, or the hidden layer activation function
    nn::Model model({
        nn::layers::fully_connected(4),
        nn::layers::relu(),
        nn::layers::fully_connected(1),
        nn::layers::sigmoid()
    });

    // build the model according to a single input shape (without the batch dimension)
    // in this case, each input contains two values: the two bits in the xor operation
    model.build({2});

    // print the summary of the model - see the output at the end 
    model.summary();

    // compile the model with an 'adam' optimizer,
    // 'binary_crossentropy' loss function and
    // 'binary_accuracy' metric.
    // different optimizer configurations can be
    // tested for different results
    model.compile(nn::optimizers::adam(),
    nn::losses::binary_crossentropy("loss"),
    {nn::metrics::binary_accuracy("accuracy")});

    // train the current model and retrieve it's 'History'
    // object (currently unused) with the following
    // configurations:
    // - inputs to train on: x
    // - labels to train on: y
    // - callbacks: none
    // - batch size: 32
    // - epochs: 1000
    auto train_history = model.fit(x, y, {}, 32, 1000);

    // test the current model and retrieve it's 'History'
    // object (currently unused) with the following
    // configurations:
    // - inputs to test on: x
    // - batch size: 32
    // - verbose: 0
    auto eval_history = model.evaluate(x, y);

    // print the prediction of the model on all
    // 4 xor cases - see output at the end
    std::cout << model.predict(x, 32, 0) << std::endl;

    // save the weights of the model in the
    // specified file path (relative path)
    model.save_weights("examples/xor/xor_weights");

    return 0;

    // program compiled with:
    // g++ -O3 -g -I. -std=c++17 examples/xor/xor_model.cc -o xor_test
    
    /*
        program output:

        Layer Type        Output Shape        Params
        --------------------------------------------
        Fully Connected   (None, 4)           12
        ReLU              (None, 4)           0
        Fully Connected   (None, 1)           5
        Sigmoid           (None, 1)           0

        Total Params: 17
        --------------------------------------------
        epoch 1/1000 - 1/1 - [====================] - loss: 0.6998 - binary_accuracy: 0.5000
        epoch 2/1000 - 1/1 - [====================] - loss: 0.6994 - binary_accuracy: 0.5000
        epoch 3/1000 - 1/1 - [====================] - loss: 0.6989 - binary_accuracy: 0.5000
        .
        .
        .
        epoch 998/1000 - 1/1 - [====================] - loss: 0.2848 - binary_accuracy: 1.0000
        epoch 999/1000 - 1/1 - [====================] - loss: 0.2844 - binary_accuracy: 1.0000
        epoch 1000/1000 - 1/1 - [====================] - loss: 0.2840 - binary_accuracy: 1.0000
        epoch 1/1 - 1/1 - [====================] - loss: 0.2835 - binary_accuracy: 1.0000
        [[0.297512]
        [0.792283]
        [0.751502]
        [0.230993]]
    */

    // note: not all training output is shown, epochs 4 - 996 (inclusive)
    // have been removed
}