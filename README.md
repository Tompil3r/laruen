![Logo with padding](https://user-images.githubusercontent.com/67478051/192153302-2a956551-233a-4a5f-ade4-ae08cbbeb869.png)
---
# About Laruen

Laruen is a mathematical and deep learning library written in C++. Laruen provides a simple api, yet efficient implementation,
allowing for a fast and straightforward neural network training and various multi dimensional computations.
With Laruen's arsenal of objects, functions and tools you will easily be able to experiment and create
many powerful ideas. Laruen is implemented entirely in header files, removing the headache of compiling the
necessary files.

---

# Introduction to Laruen
## Mathematics
The core structure for the mathematical part of Laruen is Laruen's multi dimensional array called `NDArray`.
`NDArray` functions similarly to the python's package - numpy, ndarray object.
The basic properties of Laruen's `NDArray` are:
- Provides a template argument for all arithmetic types.
- Allows broadcasting of arrays with different number of dimensions.
- Provides efficient implementation for a large number of mathematical functions, including Strassen-Winograd implementation of matrix multiplication.
- NDArray works with non contiguous data, allowing for slicing in all dimensions.

The NDArray class is residing within the `laruen::multi` namespace, in order to include `NDArray` and other necessary types use:
```cpp
#include <laruen/multi/ndarray.h>
#include <laruen/multi/types.h>
using namespace laruen::multi;
```

Creating an `NDArray`:
```cpp
NDArray<int> array(Shape{2, 3, 4});
```

---

## Neural Networks
The core structure for the machine learning part of Laruen is its abstract Neural Network model, simply called `Model`.
Laruen's model structure allows for flexible and easy creation, training and testing of many different neural network models.
The abstract `Layer` structure is the main property of the `Model` class. Each model may contain several types of layers,
to name a few: `FullyConnected`, `ReLU` and `Softmax`.
With that, every model will have a loss function, that will evaluate the performance of the neural network on a given problem.
Lauren provides a variety of loss functions that should be compatible with the vast majority of problems.
In addition, Laruen also has a large number of:
- Optimizers
- Initializers (weight initializers)
- Callbacks
- Metrics
- Other useful utility functions

Fun fact: If you have ever worked with the python library Keras, you may notice that Laruen's api is very similar to Keras`.
As a Keras user myself, I found its api to be very simple and easy to work with, and for that reason Laruen adapted a similar one.

The `Model` class is located inside the `laruen::nn` namespace, with each individual component (like `Layer`s and `Metric`s) in their own sub namespace, e.g:
`laruen::nn::layers` and `laruen::nn::metrics`. In order to include the `Model` class use the following:
```cpp
#include <laruen/nn/model.h>
using laruen::nn::Model;
```

Here is a snippet of code to create a simple neural network:
```cpp
#include <laruen/nn/model.h>
#include <laruen/nn/layers/fully_connected.h>
#include <laruen/nn/layers/sigmoid.h>

int main() {
  using laruen::nn::Model;
  using laruen::nn::layers::fully_connected, laruen::nn::layers::sigmoid;
  
  Model<> model({
    fully_connected(32),  // first layer - a fully connected layer with 32 nodes
    sigmoid(),            // second layer - a sigmoid activation function
    fully_connected(1),   // third layer - a fully connected layer with 1 node
    sigmoid()             // fourth layer - another sigmoid activation function
  });
  
  return 0;
}
```

Now, to initialize the weights of the neural network in the correct dimensions use the `.build()` method.
The `build` method requires a `Shape` of one **single** input as an argument (without the samples / batch dimension).
```cpp
model.build({10}); // in this example each input will have 10 values
```

Once your model's structure is complete, configure its learning parameters with the `.compile()` method.
The `compile` method requires an `Optimizer` to update the weights, a `Loss` class (function) to evaluate the performance of the neural network
and optionally a list of `Metric`s which will be displayed during the training and evaluation processes.
In this example, a problem of binary classification will be used to determine the `Loss` function and the `Metric`s:
```cpp
model.compile(
  adam(),
  binary_crossentropy(),
  {binary_accuracy()}
);
```

To start the training process use the `.fit()` method:
```cpp
// x_train and y_train are NDArrays with data to train on
// in this example we are training with a list of empty Callbacks, a batch size of 32 and for a total of 3 epochs
auto training_history = model.fit(x_train, y_train, {}, 32, 3);
```

If you want to evaluate the performace of your model when training is done, use the `.evaluate()` method:
```cpp
// x_test and y_test are NDArrays with data to test on
auto evaluation_history = model.evaluate(x_test, y_test);
```

And finally, to make predictions on samples of data, use the `.predict()` method:
```cpp
auto predictions = model.predict(x_test);
```

If you need more explanation and examples, please see: https://github.com/Tompil3r/laruen/tree/main/examples, which has two very nice examples 
with different complexities for building neural networks with Laruen.
- The first example is a neural network that attempts to calculate the binary xor function.
- The second example is a neural network whose purpose is to predict handwritten digits (using the MNIST handwritten digits dataset).

---

# Issues and features
If you would like to report a bug or suggest a feature use the [Github issues](https://github.com/Tompil3r/laruen/issues) page.
