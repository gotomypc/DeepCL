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
class DeepCL_EXPORT Batcher2 {
public:
    const int N;
    const int batchSize;

    const int numBatches;

    int nextBatch;
    bool epochDone;

    virtual void _tick( int batchStart, int thisBatchSize ) = 0;
    virtual void _reset() = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    Batcher2( int N, int batchSize );
    bool tick();
    void reset();
    void runEpoch( float learningRate );

    // [[[end]]]
};


