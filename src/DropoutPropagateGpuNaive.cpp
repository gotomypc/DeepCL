// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"

#include "StatefulTimer.h"
#include "stringhelper.h"

#include "DropoutPropagateGpuNaive.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL DropoutPropagateGpuNaive::~DropoutPropagateGpuNaive() {
    delete kernel;
}
VIRTUAL void DropoutPropagateGpuNaive::propagate( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
//    cout << StatefulTimer::instance()->prefix << "DropoutPropagateGpuNaive::propagate( CLWrapper * )" << endl;
    StatefulTimer::instance()->timeCheck("DropoutPropagateGpuNaive::propagate start" );

    kernel  ->input( batchSize * numPlanes * outputImageSize * outputImageSize )
            ->input( masksWrapper )
            ->input( inputWrapper )
            ->output( outputWrapper );
    int globalSize = batchSize * numPlanes * outputImageSize * outputImageSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    cout << "DropoutPropagateGpuNaive::propagate batchsize=" << batchSize << " g=" << globalSize << " w=" << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

//    cout << "DropoutPropagateGpuNaive::propagate selectorswrapper:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );

    StatefulTimer::instance()->timeCheck("DropoutPropagateGpuNaive::propagate end" );
}
DropoutPropagateGpuNaive::DropoutPropagateGpuNaive( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        DropoutPropagate( cl, numPlanes, inputImageSize, dropRatio ) {
    string options = "";
    options += " -DgOutputImageSize=" + toString( outputImageSize );
    options += " -DgOutputImageSizeSquared=" + toString( outputImageSize * outputImageSize );
    options += " -DgInputImageSize=" + toString( inputImageSize );
    options += " -DgInputImageSizeSquared=" + toString( inputImageSize * inputImageSize );
    options += " -DgNumPlanes=" + toString( numPlanes );
//    float inverseDropRatio = 1.0f / dropRatio;
//    string inverseDropRatioString = toString( inverseDropRatio );
//    if( inverseDropRatioString.find( "." ) == string::npos ) {
//        inverseDropRatioString += ".0f";
//    } else {
//        inverseDropRatioString += "f";
//    }
////    cout << "inverseDropRatioString " << inverseDropRatioString << endl;
//    options += " -D gInverseDropRatio=" + inverseDropRatioString;

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/dropout.cl", "propagateNaive", 'options' )
    // ]]]
    // generated using cog, from cl/dropout.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void propagateNaive(\n" 
    "        const int N,\n" 
    "        global const unsigned char *mask,\n" 
    "        global const float *input,\n" 
    "        global float *output ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;\n" 
    "}\n" 
    "\n" 
    "kernel void backpropNaive(\n" 
    "        const int N,\n" 
    "        global const unsigned char *mask,\n" 
    "        global const float *errors,\n" 
    "        global float *output) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    output[globalId] = mask[globalId] == 1 ? errors[globalId] : 0.0f;\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagateNaive", options, "cl/dropout.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "dropout.cl", "propagateNaive", options );
}

