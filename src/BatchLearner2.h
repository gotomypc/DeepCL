#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

class NeuralNet;
class Trainable;

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

// callback free. that's the plan (since makes easier to wrap for lua et al)
template< typename T >
class DeepCL_EXPORT BatchLearner2 {
public:
    NeuralNet *net;
    const int N;
    const int batchSize;
    T const*data;
    int const *labels;

    const int numBatches;
    const int inputCubeSize;
//    const int outputCubeSize;

    int nextBatch;
    int numRight;
    float loss;
    bool epochDone;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearner2( NeuralNet *net, int N, int batchSize, T *data, int const *labels );
    bool tick( float learningRate );
    void reset();
    void runEpoch( float learningRate );

    // [[[end]]]
};

