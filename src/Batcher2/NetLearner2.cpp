// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearner.h"
#include "NeuralNet.h"
#include "Trainable.h"

#include "NetLearner2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T > NetLearner2<T>::NetLearner2( Trainable *net ) :
        net( net ) {
    batchSize = 128;
    learningRate = 0.002f;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    startEpoch = 1;
    dumpTimings = false;
    batchLearner = 0;
}

template< typename T > void NetLearner2<T>::setTrainingData( int Ntrain, T *trainData, int *trainLabels ) {
    this->Ntrain = Ntrain;
    this->trainData = trainData;
    this->trainLabels = trainLabels;
}

template< typename T > void NetLearner2<T>::setTestingData( int Ntest, T *testData, int *testLabels ) {
    this->Ntest = Ntest;
    this->testData = testData;
    this->testLabels = testLabels;
}

template< typename T > void NetLearner2<T>::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 1 );
}

template< typename T > void NetLearner2<T>::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

template< typename T > void NetLearner2<T>::setSchedule( int numEpochs, int startEpoch ) {
    this->numEpochs = numEpochs;
    this->startEpoch = startEpoch;
}

template< typename T > void NetLearner2<T>::setBatchSize( int batchSize ) {
    this->batchSize = batchSize;
}

template< typename T > void NetLearner2<T>::learningRate( float learningRate ) {
    this->learningRate = learningRate;
}

template< typename T > VIRTUAL NetLearner2<T>::~NetLearner2() {
    delete batchLearner;
}

template< typename T > void NetLearner2<T>::tickEpoch() {
}

template< typename T > void NetLearner2<T>::tickBatch() {
}

template< typename T > void NetLearner2<T>::learn() {
    Timer timer;
    for( int epoch = startEpoch; epoch <= numEpochs; epoch++ ) {
        float annealedLearningRate = learningRate * pow( annealLearningRate, epoch );
        EpochResult epochResult = batchLearner.runEpochFromLabels( annealedLearningRate, batchSize, Ntrain, trainData, trainLabels );
        if( dumpTimings ) {
            StatefulTimer::dump(true);
        }
//        cout << "-----------------------" << endl;
        cout << endl;
        timer.timeCheck("after epoch " + toString(epoch ) );
        cout << "annealed learning rate: " << annealedLearningRate << " training loss: " << epochResult.loss << endl;
        cout << " train accuracy: " << epochResult.numRight << "/" << Ntrain << " " << (epochResult.numRight * 100.0f/ Ntrain) << "%" << std::endl;
        int testNumRight = batchLearner.test( batchSize, Ntest, testData, testLabels );
        cout << "test accuracy: " << testNumRight << "/" << Ntest << " " << (testNumRight * 100.0f / Ntest ) << "%" << endl;
        timer.timeCheck("after tests");
    }
}

template class NetLearner2<unsigned char>;
template class NetLearner2<float>;

