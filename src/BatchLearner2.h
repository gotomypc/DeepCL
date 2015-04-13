#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

class NeuralNet;
class Trainable;

#include "Batcher2.h"

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

// callback free. that's the plan (since makes easier to wrap for lua et al)
template< typename T >
class DeepCL_EXPORT BatchLearner2 : public Batcher2<T> {
public:
    NeuralNet *net;
    const float learningRate;
    T const*data;
    int const *labels;

    const int inputCubeSize;
//    const int outputCubeSize;

    int numRight;
    float loss;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearner2( int N, int batchSize, NeuralNet *net, float learningRate,
    T *data, int const *labels
    );
    void _tick(int batchStart, int thisBatchSize);
    void _reset();

    // [[[end]]]
};

