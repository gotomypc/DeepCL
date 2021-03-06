#pragma once

#include <string>
#include <stdexcept>
#include <iostream>

extern int exceptionRaised;
extern std::string exceptionMessage;
void raiseException( std::string message );
void checkException( int *wasRaised, std::string *message );

#include "NetLearner.h"

// we need this, so we can catch the c++ exception, and raise
// it in our altenrative way, all without needing to use the gil
// (which I *think* adding 'except +' requires?)
class CyNetLearner : public NetLearner {
public:
    CyNetLearner(Trainable *neuralNet,
            int Ntrain, float *trainData, int *trainLabels,
            int Ntest, float *testData, int *testLabels,
            int batchSize ) :
        NetLearner( neuralNet,
            Ntrain, trainData, trainLabels,
            Ntest, testData, testLabels,
            batchSize ) {
    }
    void learn( float learningRate ) {
        try {
            NetLearner::learn(learningRate);
        } catch( std::runtime_error &e ) {
            std::cout << e.what() << std::endl;
            raiseException( e.what() );
        }
    }
};

