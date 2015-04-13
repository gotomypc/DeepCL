// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "Trainable.h"

#include "BatchPropagator2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T >
BatchPropagator2<T>::BatchPropagator2( int N, int batchSize, NeuralNet *net,
        T *data
     ) :
        Batcher2( N, batchSize ),
        net( net ),
        data( data ),
        inputCubeSize( net->getInputCubeSize() ) {
}

// do one batch, update variables
// returns true if not finished, otherwise false
template< typename T >
void BatchPropagator2<T>::_tick(int batchStart, int thisBatchSize) {
    net->setBatchSize( thisBatchSize );
    net->propagate( &(data[ batchStart * inputCubeSize ]) );
 }

template< typename T >
void BatchPropagator2<T>::_reset() {
}

template class BatchPropagator2<unsigned char>;
template class BatchPropagator2<float>;

