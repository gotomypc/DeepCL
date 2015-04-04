# Development

## Development technical details

* [cogapp](http://nedbatchelder.com/code/cog/) generator is used extensively, to accelerate development, reduce the number of manual copy-and-pasting and so on.  Specifically, it's used for:
  * generating header declarations from .cpp definition files
  * generating fluent-style argument classes for certain tests
  * ... and more uses will surely be found :-)
* You need Python installed and available for this to work.  You don't need python just to
build the sources, but if you do have python installed, and you flip the `PYTHON_AVAILABLE` switch in the 
cmake configuration, then a lot of manual editing will no longer be necessary :-)

## Architecture

* [NeuralNet.h](src/NeuralNet.h) is a container for layers. It contains three types of method:
  * methods that iterate over each layer, eg `propagate`
  * methods that call a method on the first layer, eg `getInputCubeSize`
  * methods that call a method on the last layer, eg `getResults()`
* Various net layers, eg [ConvolutionalLayer.cpp](src/ConvolutionalLayer.cpp), [PoolingLayer.cpp](src/PoolingLayer.cpp), etc
* Trying to debug/unit-test by training whole layers is challenging, so the layer implementations are factorized, over two levels.  The first level abstracts away propagation, backprop of errors, and backprop of weights:
  * [Propagate.cpp](src/Propagate.cpp) handles forward propagation
  * [BackpropErrorsv2.cpp](src/BackpropErrorsv2.cpp) handles backward propagation of errors (strictly speaking: of the partial derivative of the loss with respect to the pre-activation sums for the layer)
    * The results of this layer are passed back through the stack of layers
  * [BackpropWeights2.cpp](src/BackpropWeights2.cpp) handles backward propagation of weights, from the results of the appropriate BackpropErrorsv2 layer
* Then, each of these classes calls into implementation classes, which are children of the same class, which provide various kernels and implementations.  Eg, for [Propagate.h](src/Propagate.h], we have:
  * [Propagate1.cpp](src/Propagate1.cpp)
  * [Propagate2.cpp](src/Propagate2.cpp)
  * [Propagate3.cpp](src/Propagate3.cpp)
  * ...
* ... and similarly for [BackpropErrorsv2](src/BackpropErrorsv2.cpp), and [BackpropWeights2.cpp](src/BackpropWeights2.cpp): each has implementation classes
* Therefore:
  * Testing can target one single implementation, or target only propagate or backproperrors, or backpropweights, rather than needing to test an entire network
  * These lower level factorized implementations could also plausibly be an appropriate unit of re-use
* There are also "meta"-layers, ie:
  * [PropagateAuto.cpp](src/PropagateAuto.cpp): automatically tries different propagate kernels at run-time, and chooses the fastest :-)

## Testing

### Correctness checking

* For forward propagation:
  * We slot in some numbers, calculate the results manually, and compare with results actually obtained
  * We also forward propagate pictures/photos, and check the results look approximately like what we would expect
* For backward propagation:
  * We use numerical validation, since the sum of the square of the weight changes, divided by the learning rate, approximately equals the change in loss.  Or it should. We test this :-)
* Standard test sets
  * Checked using implementations for MNIST, and NORB is in progress

### Concepts

* Network optimization is stochastic, and there are typically numerous local minima, into which the optimization can get stuck
* For unit testing, this is not very suitable, since unit tests must run repeatably, reliably, quickly
* Therefore, for unit-testing, the network weights are preset to a fixed set of values
  * using a random number generator with a fixed seed
  * or by explicitly giving a hard-coded set of weights
* Then, the test checks that the network converges to better than an expected loss, and accuracy, within a preset number of epochs
* We also have unit tests for forward-propagation, and backward propagation, as per section [Correctness checking](#correctness-checking) above.

### Implementation

* Using googletest, which:
  * compiles quickly
  * gives awesome colored output
  * lets you choose which tests to run using `--gtest_filter=` option
* Dont need to install anything: it's included in the `thirdparty` directory, and added to the build automatically
* To run the unit tests:
```bash
make unittests
./unittests
```
* To run just the unittests for eg `testbackprop`, do:
```bash
make unittests
./unittests --gtest_filter=testbackprop.*
```
* To skip any slow tests, do:
```bash
./unittests --gtest_filter=-*SLOW*
```
* Actually, by default, with no arguments, the argument `--gtest_filter=-SLOW*` will be appended automatically
* Also, rather than having to type `--gtest_filter=[something]`, you can just type `tests=[something]`, and this will be converted into `--gtest_filter=[something]` automatically

## Third-party libraries

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)


