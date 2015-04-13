// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "Trainable.h"

#include "BatchLearner2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T >
BatchLearner2<T>::BatchLearner2( NeuralNet *net, int N, int batchSize, T *data, int const *labels ) :
        net( net ),
        N( N ),
        batchSize( batchSize ),
        data( data ),
        labels( labels ),
        numBatches( ( N + batchSize  - 1 ) / batchSize ),
        inputCubeSize( net->getInputCubeSize() ) {
    nextBatch = 0;
    numRight = 0;
    loss = 0;
    epochDone = false;
}

// do one batch, update variables
// returns true if not finished, otherwise false
template< typename T >
bool BatchLearner2<T>::tick( float learningRate ) {
    if( epochDone ) {
        epochDone = false;
    }
    int batch = nextBatch;
    int batchStart = batch * batchSize;
    int thisBatchSize = batchSize;
    if( batch == numBatches - 1 ) {
        thisBatchSize = N - batchStart;
    }
//    cout << "numbatches " << numBatches << endl;
    net->setBatchSize( thisBatchSize );
    net->learnBatchFromLabels( learningRate, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
    loss += net->calcLossFromLabels( &(labels[batchStart]) );
    numRight += net->calcNumRight( &(labels[batchStart]) );

    nextBatch++;
    if( nextBatch == numBatches ) {
        epochDone = true;
//        nextBatch = 0;
    }
    return !epochDone;
}

template< typename T >
void BatchLearner2<T>::reset() {
    epochDone = false;
    nextBatch = 0;
    loss = 0;
    numRight = 0;
}

template< typename T >
void BatchLearner2<T>::runEpoch( float learningRate ) {
    while( !epochDone ) {
        tick( learningRate );
    }
}

template class BatchLearner2<unsigned char>;
template class BatchLearner2<float>;

