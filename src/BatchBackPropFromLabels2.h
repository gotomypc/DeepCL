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
class DeepCL_EXPORT BatchBackPropFromLabels2 : public Batcher2 {
public:
    NeuralNet *net;
    const float learningRate;
    int const *labels;

    const int inputCubeSize;

    int numRight;
    float loss;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchBackPropFromLabels2( int N, int batchSize, NeuralNet *net, float learningRate,
    int const *labels
    );
    void _tick(int batchStart, int thisBatchSize);
    void _reset();

    // [[[end]]]
};

