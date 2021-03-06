// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "Timer.h"
#include "ConvolutionalLayer.h"
#include "LayerMaker.h"
#include "ActivationFunction.h"
#include "StatefulTimer.h"
#include "AccuracyHelper.h"
#include "NeuralNetMould.h"
#include "Layer.h"
#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "EpochMaker.h"
#include "LossLayer.h"
#include "IAcceptsLabels.h"
#include "ExceptionMacros.h"
#include "InputLayerMaker.h"

#include "NeuralNet.h"

using namespace std;

//static std::mt19937 random;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

PUBLICAPI NeuralNet::NeuralNet() :
    isTraining( true ) {
    cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
}
NeuralNet::NeuralNet( int gpu ) :
    isTraining( true ) {

    cl = OpenCLHelper::createForIndexedGpu( gpu );
}
/// Constructor
PUBLICAPI NeuralNet::NeuralNet( int numPlanes, int imageSize ) {
//    cout << "NeuralNet( " << numPlanes << ", " << imageSize << " )" << endl;
    cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    addLayer( InputLayerMaker::instance()->numPlanes( numPlanes )->imageSize( imageSize ) );
//    print();
}
NeuralNet::~NeuralNet() {
    for( int i = 0; i < (int)layers.size(); i++ ) {
        delete layers[i];
    }
    delete cl;
}
NeuralNet *NeuralNet::clone() {
    NeuralNet *copy = new NeuralNet();
    for( vector<Layer *>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        LayerMaker2 *maker = (*it)->maker;

        LayerMaker2 *makerCopy = maker->clone();
        copy->addLayer( makerCopy );
    }
    copy->print();
    cout << "outputimagesize: " << copy->getOutputImageSize() << endl;
    return copy;
}
OpenCLHelper *NeuralNet::getCl() {
    return cl;
}
STATIC NeuralNetMould *NeuralNet::maker() {
    return new NeuralNetMould();
}

/// Add a network layer, using a LayerMaker2 object
PUBLICAPI void NeuralNet::addLayer( LayerMaker2 *maker ) {
//    cout << "neuralnet::insert numplanes " << inputLayerMaker._numPlanes << " imageSize " << inputLayerMaker._imageSize << endl;
    maker->setCl( cl );
    Layer *layer = maker->createLayer( getLastLayer() );
    layers.push_back( layer );
}
PUBLICAPI void NeuralNet::initWeights( int layerIndex, float *weights, float *biasWeights ) {
    initWeights( layerIndex, weights );
    initBiasWeights( layerIndex, biasWeights );
}
PUBLICAPI void NeuralNet::initWeights( int layerIndex, float *weights ) {
    layers[layerIndex]->initWeights( weights );
}
PUBLICAPI void NeuralNet::initBiasWeights( int layerIndex, float *weights ) {
    layers[layerIndex]->initBiasWeights( weights );
}
void NeuralNet::printWeightsAsCode() {
    for( int layer = 1; layer < (int)layers.size(); layer++ ) {
        layers[layer]->printWeightsAsCode();
    }
}
void NeuralNet::printBiasWeightsAsCode() {
    for( int layer = 1; layer < (int)layers.size(); layer++ ) {
        layers[layer]->printBiasWeightsAsCode();
    }
}
/// \brief calculate the loss, based on the passed in expectedValues array
///
/// \publicapi
///
/// Calculate the loss, based on the passed in expectedValues array
/// which should be the same size as the results of the final layer
/// of the network
PUBLICAPI float NeuralNet::calcLoss(float const *expectedValues ) {
    return dynamic_cast<LossLayer*>(layers[layers.size()-1])->calcLoss( expectedValues );
}
PUBLICAPI float NeuralNet::calcLossFromLabels(int const *labels ) {
    return dynamic_cast<IAcceptsLabels*>(layers[layers.size()-1])->calcLossFromLabels( labels );
}
/** \brief **/EpochMaker *NeuralNet::epochMaker() {
     return new EpochMaker(this);
}
VIRTUAL LossLayerMaker *NeuralNet::cloneLossLayerMaker() const {
    LossLayer const *lossLayer = dynamic_cast< LossLayer const*>( getLastLayer() );
    if( lossLayer == 0 ) {
        throw runtime_error("error: last layer must be a losslayer");
    }
    return dynamic_cast< LossLayerMaker *>( lossLayer->maker->clone() );
//    throw runtime_error("need to implement neuralnet::clonelosslayermaker :-)" );
//    LossLayer const*lossLayer = dynamic_cast< LossLayer const*>( getLastLayer() );
//    return dynamic_cast< LossLayerMaker *>( lossLayer->maker->clone( clonePreviousLayer ) ) ;
}
PUBLICAPI InputLayer *NeuralNet::getFirstLayer() {
    return dynamic_cast<InputLayer *>( layers[0] );
}
PUBLICAPI Layer *NeuralNet::getLastLayer() {
    if( layers.size() == 0 ) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI int NeuralNet::getNumLayers() const {
    return (int)layers.size();
}
PUBLICAPI Layer *NeuralNet::getLayer( int index ) {
    if( layers.size() == 0 ) {
        return 0;
    }
    if( index < 0 || index > (int)layers.size() - 1 ) {
        return 0;
    }
    return layers[index];
}
PUBLICAPI Layer const*NeuralNet::getLastLayer() const {
    if( layers.size() == 0 ) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputPlanes() const {
    return getLastLayer()->getOutputPlanes();
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputImageSize() const {
    return getLastLayer()->getOutputImageSize();
}
//Layer *NeuralNet::addLayer( LayerMaker *maker ) {
////    Layer *previousLayer = 0;
////    if( layers.size() > 0 ) {
////        previousLayer = layers[ layers.size() - 1 ];
////    }
////    maker->setPreviousLayer( previousLayer );
//    Layer *layer = maker->instance();
//    layers.push_back( layer );
//    return layer;
//}
PUBLICAPI void NeuralNet::setBatchSize( int batchSize ) {
    for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        (*it)->setBatchSize( batchSize );
    }
}
PUBLICAPI void NeuralNet::setTraining( bool training ) {
    for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        (*it)->setTraining( training );
    }
}
PUBLICAPI int NeuralNet::calcNumRight( int const *labels ) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if( acceptsLabels == 0 ) {
        THROW("You need to add a IAcceptsLabels as the last layer, in order to use calcNumRight");
    }
    return acceptsLabels->calcNumRight( labels );
}
PUBLICAPI void NeuralNet::propagate( float const*images) {
    // forward...
    dynamic_cast<InputLayer *>(layers[0])->in( images );
    for( int layerId = 0; layerId < (int)layers.size(); layerId++ ) {
        StatefulTimer::setPrefix("layer" + toString(layerId) + " " );
        layers[layerId]->propagate();
        StatefulTimer::setPrefix("" );
    }
}
PUBLICAPI void NeuralNet::backPropFromLabels( float learningRate, int const *labels) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if( acceptsLabels == 0 ) {
        throw std::runtime_error("Must add a child of IAcceptsLabels as last layer, to use backPropFromLabels");
    }
    acceptsLabels->calcErrorsFromLabels( labels );
    for( int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " " );
        Layer *layer = layers[layerIdx];
        if( layer->needsBackProp() ) {
            layer->backProp( learningRate );
        }
        StatefulTimer::setPrefix("" );
    }
}
PUBLICAPI void NeuralNet::backProp( float learningRate, float const *expectedResults) {
    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    if( lossLayer == 0 ) {
        throw std::runtime_error("Must add a LossLayer as last layer of net");
    }
    lossLayer->calcErrors( expectedResults );
    for( int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " " );
        layers[layerIdx]->backProp( learningRate );
        StatefulTimer::setPrefix("" );
    }
}
PUBLICAPI int NeuralNet::getNumLayers() {
    return (int)layers.size();
}
PUBLICAPI float const *NeuralNet::getResults( int layer ) const {
    return layers[layer]->getResults();
}
PUBLICAPI int NeuralNet::getInputCubeSize() const {
    return layers[ 0 ]->getOutputCubeSize();
}
PUBLICAPI int NeuralNet::getOutputCubeSize() const {
    return layers[ layers.size() - 1 ]->getOutputCubeSize();
}
PUBLICAPI float const *NeuralNet::getResults() const {
    return getResults( (int)layers.size() - 1 );
}
PUBLICAPI VIRTUAL int NeuralNet::getResultsSize() const {
    return getLastLayer()->getResultsSize();
}
void NeuralNet::print() {
    cout << this->asString();
//    int i = 0; 
//    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
//        std::cout << "layer " << i << ":" << (*it)->asString() << endl;
//        i++;
//    }
}
void NeuralNet::printWeights() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printWeights();
        i++;
    }
}
void NeuralNet::printOutput() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printOutput();
        i++;
    }
}
PUBLICAPI std::string NeuralNet::asString() {
    std::string result = "";
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        result += "layer " + toString( i ) + ":" + (*it)->asString() + "\n";
        i++;
    }    
    return result;
}

//template DeepCL_EXPORT InputLayerMaker<unsigned char> *NeuralNet::inputMaker<unsigned char>();
//template DeepCL_EXPORT InputLayerMaker<float> *NeuralNet::inputMaker<float>();

//template DeepCL_EXPORT void NeuralNet::append( InputLayerMaker<unsigned char> inputLayerMaker );
//template DeepCL_EXPORT void NeuralNet::append( InputLayerMaker<float> inputLayerMaker );

