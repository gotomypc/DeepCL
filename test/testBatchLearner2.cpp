#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

#include "NeuralNet.h"
//#include "WeightsPersister.h"
#include "NetdefToNet.h"
#include "GenericLoader.h"
#include "BatchLearner2.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "test/TestArgsParser.h"
#include "Timer.h"

using namespace std;

TEST( testBatchLearner2, basic ) {
    string dataDir = "../data/mnist";
    string trainFile = "train-images-idx3-ubyte";
    string testFile = "t10k-images-idx3-ubyte";
    const int Ntrain = 1280;
    const int Ntest = 1280;
    const int batchSize = 128;
    const float learningRate = 0.002f;
    const float translate = -32.7936;
    const float scale = 0.00643144;

    int numPlanes;
    int imageSize;

    unsigned char *trainData = 0;
    unsigned char *testData = 0;
    int *trainLabels = 0;
    int *testLabels = 0;

//    GenericLoader::getDimensions( dataDir + "/" + trainFile, &Ntrain, &numPlanes, &imageSize );
    numPlanes = 1;
    imageSize = 28;
    cout << "Ntrain " << Ntrain << " numPlanes " << numPlanes << " imageSize " << imageSize << endl;
    trainData = new unsigned char[ (long)Ntrain * numPlanes * imageSize * imageSize ];
    trainLabels = new int[Ntrain];
    GenericLoader::load( dataDir + "/" + trainFile, trainData, trainLabels, 0, Ntrain );

    testData = new unsigned char[ (long)Ntest * numPlanes * imageSize * imageSize ];
    testLabels = new int[Ntest];
    GenericLoader::load( dataDir + "/" + testFile, testData, testLabels, 0, Ntest );
   
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    net->addLayer( NormalizationLayerMaker::instance()->translate(translate)->scale(scale) );
    NetdefToNet::createNetFromNetdef( net, "RT2-8C5{z}-MP2-16C5{z}-MP3-150N-10N" );
    net->print();

    BatchLearner2<unsigned char> batchLearner( Ntrain, batchSize, net, learningRate, trainData, trainLabels );
    int totalBatches = 0;
    for( int epoch = 0; epoch < 3; epoch++ ) {
        batchLearner.reset();
        while( !batchLearner.epochDone ) {
            batchLearner.tick();
            cout << "epoch=" << epoch << " batch=" << ( batchLearner.nextBatch - 1 ) << " numRight=" << batchLearner.numRight <<
                " loss=" << batchLearner.loss << endl;
            totalBatches++;
        }
    }
    EXPECT_EQ( 30, totalBatches );
    EXPECT_GT( batchLearner.numRight, 600 );
    EXPECT_LT( batchLearner.loss, 1800 );

    delete  net;
}


