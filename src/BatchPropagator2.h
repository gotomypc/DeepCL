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
class DeepCL_EXPORT BatchPropagator2 : public Batcher2 {
public:
    NeuralNet *net;
    T const*data;

    const int inputCubeSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchPropagator2( int N, int batchSize, NeuralNet *net,
    T *data
    );
    void _tick(int batchStart, int thisBatchSize);
    void _reset();

    // [[[end]]]
};

