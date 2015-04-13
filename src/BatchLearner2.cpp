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
BatchLearner2<T>::BatchLearner2( int N, int batchSize, NeuralNet *net, float learningRate,
        T *data, int const *labels
     ) :
        Batcher2<T>( N, batchSize ),
        net( net ),
        learningRate( learningRate ),
        data( data ),
        labels( labels ),
        inputCubeSize( net->getInputCubeSize() ) {
    numRight = 0;
    loss = 0;
}

// do one batch, update variables
// returns true if not finished, otherwise false
template< typename T >
void BatchLearner2<T>::_tick(int batchStart, int thisBatchSize) {
    net->setBatchSize( thisBatchSize );
    net->learnBatchFromLabels( learningRate, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
    loss += net->calcLossFromLabels( &(labels[batchStart]) );
    numRight += net->calcNumRight( &(labels[batchStart]) );
 }

template< typename T >
void BatchLearner2<T>::_reset() {
    loss = 0;
    numRight = 0;
}

template class BatchLearner2<unsigned char>;
template class BatchLearner2<float>;

