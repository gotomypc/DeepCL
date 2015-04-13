// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "Trainable.h"

#include "Batcher2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

Batcher2::Batcher2( int N, int batchSize ) :
        N( N ),
        batchSize( batchSize ),
        numBatches( ( N + batchSize  - 1 ) / batchSize ) {
    nextBatch = 0;
    epochDone = false;
}

// do one batch, update variables
// returns true if not finished, otherwise false
bool Batcher2::tick() {
    if( epochDone ) {
        epochDone = false;
    }
    int batch = nextBatch;
    int batchStart = batch * batchSize;
    int thisBatchSize = batchSize;
    if( batch == numBatches - 1 ) {
        thisBatchSize = N - batchStart;
    }

    this->_tick(batchStart, thisBatchSize);

    nextBatch++;
    if( nextBatch == numBatches ) {
        epochDone = true;
    }
    return !epochDone;
}

void Batcher2::reset() {
    epochDone = false;
    nextBatch = 0;
    this->_reset();
}

void Batcher2::runEpoch( float learningRate ) {
    while( !epochDone ) {
        tick();
    }
}


