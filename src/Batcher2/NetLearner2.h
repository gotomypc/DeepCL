// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#define VIRTUAL virtual
#define STATIC static

class NeuralNet;
class Trainable;
class BatchLearner2;

#include "DeepCLDllExport.h"

class DeepCL_EXPORT PostEpochAction {
public:
    virtual void run( int epoch ) = 0;
};
class DeepCL_EXPORT NetLearner2_PostBatchAction {
public:
    virtual void run( int epoch, int batch, float lossSoFar, int numRightSoFar ) = 0;
};

// handles learning the neural net, ie running multiple epochs,
// using a BatchLearner, to learn each epoch
template<typename T>
class DeepCL_EXPORT NetLearner2 {
public:
    Trainable *net;

    int Ntrain;
    int Ntest;
    T *trainData;
    int *trainLabels;
    T *testData;
    int *testLabels;

    int batchSize;

    float learningRate;
    float annealLearningRate;

    bool dumpTimings;

    int startEpoch;
    int numEpochs;

    BatchLearner2<T> *batchLearner;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    NetLearner2( Trainable *net );
    void setTrainingData( int Ntrain, T *trainData, int *trainLabels );
    void setTestingData( int Ntest, T *testData, int *testLabels );
    void setSchedule( int numEpochs );
    void setDumpTimings( bool dumpTimings );
    void setSchedule( int numEpochs, int startEpoch );
    void setBatchSize( int batchSize );
    void learningRate( float learningRate );
    VIRTUAL ~NetLearner2();
    void tickEpoch();
    void tickBatch();
    void learn();

    // [[[end]]]
};


