// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "ActivationBackprop.h"
#include "StatefulTimer.h"
#include "stringhelper.h"
#include "ActivationFunction.h"

#include "ActivationBackpropGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL ActivationBackpropGpuNaive::~ActivationBackpropGpuNaive() {
    delete kernel;
//    delete kMemset;
}
VIRTUAL void ActivationBackpropGpuNaive::backpropErrors( int batchSize, CLWrapper *inputWrapper,
         CLWrapper *errorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {

    StatefulTimer::instance()->timeCheck("ActivationBackpropGpuNaive::backpropErrors start" );

    int globalSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->in( batchSize * numPlanes * inputImageSize * inputImageSize )
          ->in( inputWrapper )
          ->in( errorsWrapper )
          ->out( errorsForUpstreamWrapper );
    globalSize = batchSize * numPlanes * outputImageSize * outputImageSize;
    workgroupSize = 64;
    numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

    StatefulTimer::instance()->timeCheck("ActivationBackpropGpuNaive::backpropErrors end" );
}
ActivationBackpropGpuNaive::ActivationBackpropGpuNaive( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        ActivationBackprop( cl, numPlanes, inputImageSize, fn ) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
    options += " -D gNumPlanes=" + toString( numPlanes );
    options += " -D gInputImageSize=" + toString( inputImageSize );
    options += " -D gInputImageSizeSquared=" + toString( inputImageSize * inputImageSize );
    options += " -D gOutputImageSize=" + toString( outputImageSize );
    options += " -D gOutputImageSizeSquared=" + toString( outputImageSize * outputImageSize );
    options += " -D " + fn->getDefineName();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/applyActivationDeriv.cl", "backpropErrors", 'options' )
    // ]]]
    // generated using cog, from cl/applyActivationDeriv.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 201, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH ]\n" 
    "\n" 
    "#ifdef TANH\n" 
    "    #define ACTIVATION_DERIV(output) (1 - output * output)\n" 
    "#elif defined SCALEDTANH\n" 
    "    #define ACTIVATION_DERIV(output) ( 0.66667f * ( 1.7159f - 1 / 1.7159f * output * output ) )\n" 
    "#elif defined SIGMOID\n" 
    "    #define ACTIVATION_DERIV(output) (output * ( 1 - output ) )\n" 
    "#elif defined RELU\n" 
    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)\n" 
    "#elif defined LINEAR\n" 
    "    #define ACTIVATION_DERIV(output) (1.0f)\n" 
    "#endif\n" 
    "\n" 
    "//#ifdef ACTIVATION_DERIV\n" 
    "//void kernel applyActivationDeriv(\n" 
    "//        const int N,\n" 
    "//        global float *inout ) {\n" 
    "//    int globalId = get_global_id(0);\n" 
    "//    inout[globalId] = ACTIVATION_DERIV( inout[globalId] );\n" 
    "//}\n" 
    "//#endif\n" 
    "\n" 
    "#ifdef ACTIVATION_DERIV\n" 
    "void kernel applyActivationDeriv(\n" 
    "        const int N,\n" 
    "        global float *target, global const float *source ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "    if( globalId < N ) {\n" 
    "        target[globalId] *= ACTIVATION_DERIV( source[globalId] );\n" 
    "    }\n" 
    "  //  target[globalId] *= source[globalId];\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "#ifdef ACTIVATION_DERIV\n" 
    "void kernel backpropErrors(\n" 
    "        const int N,\n" 
    "        global const float *inputs,\n" 
    "        global const float *errors,\n" 
    "        global float *errorsForUpstream ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "    if( globalId < N ) {\n" 
    "        errorsForUpstream[globalId] = ACTIVATION_DERIV( inputs[globalId] ) * errors[globalId];\n" 
    "            // probably not ideal to have the output and input separate?\n" 
    "    }\n" 
    "  //  target[globalId] *= source[globalId];\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backpropErrors", options, "cl/applyActivationDeriv.cl" );
    // [[[end]]]
}

