// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "Propagate.h"
#include "ActivationFunction.h"

#include "test/myasserts.h"
#include "test/WeightRandomizer.h"
#include "test/GtestGlobals.h"
#include "test/TestArgsParser.h"
#include "test/DimFromArgs.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

void propagateWithWipe( Propagate *prop, int batchSize, LayerDimensions dim, float *inputData, float *filters, float *biases, float *results ) {
    int inputDataSize = batchSize * dim.inputCubeSize;
    CLWrapper *dataWrapper = prop->cl->wrap( inputDataSize, inputData );
    dataWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = prop->cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = prop->cl->wrap( dim.numFilters, biases );
        biasWeightsWrapper->copyToDevice();
    }

    CLWrapper *resultsWrapper = prop->cl->wrap( batchSize * dim.outputCubeSize, results );
    memset( results, 99, sizeof(float) * batchSize * dim.outputCubeSize );
    resultsWrapper->copyToDevice(); // so we can wipe it...

    StatefulTimer::timeCheck("testpropagate: after data wrapper processing");
    prop->propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper,
            resultsWrapper );
//    StatefulTimer::timeCheck("Propagate::propagate after call propagate");
    resultsWrapper->copyToHost();
//    StatefulTimer::timeCheck("Propagate::propagate after copytohost");
    delete resultsWrapper;

    delete dataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }
}

TEST( testpropagate, imagesize2_nopadzeros ) {
    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 2;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    float data[] = { 0, 0, 
                      0.5f, 0.5f,

                        13, 17,
                       -19, 2.3f,
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int resultSize = 4;
    float expectedResults[] = {
        -0.5f * 0.5f + 0.5f * 0.5f,
        0.7f * 0.5f -1.1f * 0.5f,
        (-0.5f) * (-19) + 0.5f * 2.3f,
        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
    };
    cout << "expected number of results: " << resultSize << endl;
//    int outputImageSize = 0;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    for( int i = 1; i <= 4; i++ ) {
        Propagate *propagate = Propagate::instanceSpecific( 1, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ), new LinearActivation() );
        float *results = propagate->propagate( batchSize, data, filter1, 0 );  
        for( int result = 0; result < resultSize; result++ ) {
            ASSERT_EQ( expectedResults[result], results[result] );
        }
        delete propagate;
        delete[] results;
    }

    delete cl;
}

TEST( testpropagate, DISABLED_imagesize2_nopadzeros_skip1 ) {
    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 4;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    int skip = 1;
    float data[] = { 0, 1, 3, 0, 
                    4, 0, 0, 0, 
                      0.5f, 0, 0.5f,0, 
                      0,    0, 0,   0, 

                        13, 0, 17,0, 
                        0, 0, 0, 0, 
                       -19, 0, 2.3f,0, 
                        0, 0, 0, 0, 
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int outputImageSize = ( imageSize - filterWidth ) / ( skip + 1 ) + 1;
    cout << "outputimagesize: " << outputImageSize << endl;
    int resultsSize = outputImageSize * numOutPlanes * batchSize;
    cout << "resultssize: " << resultsSize << endl;
    float expectedResults[] = {
        -2,  0,
        0, 0,

         2.8f, 0.6f,
         1.0f, 0.1f,

         0, 0,
         0,0,

         13*0.2f,17*0.2f,
         -19*0.2f, -2.3f*1.1f


    };
    cout << "expected number of results: " << resultsSize << endl;
//    int outputImageSize = 0;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    for( int i = 1; i <= 1; i++ ) {
        Propagate *propagate = Propagate::instanceSpecific( 0, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ).setSkip(1), new LinearActivation() );
        float *results = propagate->propagate( batchSize, data, filter1, 0 );  
        for( int result = 0; result < resultsSize; result++ ) {
            cout << "checking result " << result << endl;
            EXPECT_EQ( expectedResults[result], results[result] );
        }
        delete propagate;
        delete[] results;
    }
    delete cl;
}

TEST( testpropagate, imagesize2_padzeros ) {
    int batchSize = 2;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int imageSize = 2;
    int filterWidth = 2;
    int padZeros = 1;

    float data[] = { 0, 0, 
                      0.5f, 0.3f,

                        13, 17,
                       -19, 2.3f,
};

    float filter1[] = { 0, 0,
                        -0.5f, 0.4f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,

 };
    int resultSize = (imageSize + 1) * (imageSize + 1) * batchSize * numOutPlanes;
    float *expectedResults = new float[resultSize];
    for( int i = 0; i < resultSize; i++ ) {
        expectedResults[i] = -9999; // means havent provided an expectedresult.
    }

    expectedResults[0] = 0; expectedResults[1] = 0; expectedResults[2] = 0;

    expectedResults[3] = 0.5f*0.4f;
    expectedResults[4] = 0.5f*(-0.5f)+0.4f*(0.3f);
    expectedResults[5] = 0.3f * (-0.5f); 

    expectedResults[6] = 0; expectedResults[7] = 0; expectedResults[8] = 0;

    expectedResults[9] = 0; expectedResults[10] = 0; expectedResults[11] = 0;
    expectedResults[12] =(-1.1f)*0.5;
    expectedResults[13] = 0.7f * 0.5f + (-1.1f) * 0.3f;
    expectedResults[14] = 0.7f * 0.3f;

    // plane 2, filter 2 ...
    expectedResults[27] = (-1.1f*13);
    expectedResults[28] = 0.7f * 13 + (-1.1f)*17;
    expectedResults[29] = 0.7f*17;
    expectedResults[35] = 0.2f* 2.3f;

//    expectedResults[] = 0;
//    expectedResults[5] = 0;
//    expectedResults[6] = 0.3f * 0.5f;
//    expectedResults[7] = 0.2f * 0.5f;

//    expectedResults[8] = 13 * 0.5f;
//    expectedResults[9] = 17 * (-0.5f);
//    expectedResults[10] = (-19) * 0;
//    expectedResults[11] = 2.3f * 0;
// 
//    expectedResults[12] = 13 * (-1.1f);
//    expectedResults[13] = 17 * 0.7f;
//    expectedResults[14] = (-19) * 0.3f;
//    expectedResults[15] = 2.3f * 0.2f;

//        -0.5f * 0.5f + 0.5f * 0.5f,
//        0.7f * 0.5f -1.1f * 0.5f,
//        (-0.5f) * (-19) + 0.5f * 2.3f,
//        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
//    };

//    int outputImageSize = 0;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    Propagate *propagate = Propagate::instanceTest( cl, LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( batchSize, data, filter1, 0 );        

//    ASSERT_EQ( -0.5f * 0.5f + 0.5f * 0.5f, results[0] );
//    ASSERT_EQ( 0.7f * 0.5f -1.1f * 0.5f, results[1] );
//    ASSERT_EQ( (-0.5f) * (-19) + 0.5f * 2.3f, results[2] );
//    ASSERT_EQ( 0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f , results[3] );

    for( int result = 0; result < resultSize; result++ ) {
        if( expectedResults[result] != -9999 ) {
            cout << " checking result[" << result << "]=" << results[result] << " expecting: " << expectedResults[result] << endl;
            ASSERT_FLOAT_EQ( expectedResults[result], results[result] );
        }
    }
    delete propagate;
    delete[] results;
    delete cl;
}

TEST( testpropagate, imagesize3 ) {
    int batchSize = 5;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int imageSize = 3;
    int filterWidth = 3;
    int padZeros = 0;

    float data[] = { 0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0.5f,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, -0.5f ,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0,

                        0, 0, 0,
                       0, 0, 0,
                       1, 10, 0,

                        0, 0, 0,
                       0, 0, 0,
                       0, 0, 1 
};

    float filter1[] = { 0, 0, 0,
                          0, 0, 0,
                         -0.5f, 0, 0.5f,

                        0, 0, 0,
                          0, 0, 0,
                         2.0f, 0.5, 0.5f,

 };

//    int outputImageSize = 0;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    Propagate *propagate = Propagate::instanceTest( cl, LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( 
        batchSize, data, filter1, 0 );        

    assertEquals( 0, results[0] );
    assertEquals( 1.25f, results[1] );
    assertEquals( -0.5f, results[2] );
    assertEquals( 0.75f, results[3] );
    assertEquals( -0.25f, results[4] );
    assertEquals( 1, results[5] );
    assertEquals( -0.5f, results[6] );
    assertEquals( 7, results[7] );
    assertEquals( 0.5f, results[8] );
    assertEquals( 0.5f, results[9] );
        cout << "test1 ok" << endl;
    delete propagate;
    delete[] results;
    delete cl;
}

TEST( testpropagate, test2 ) {
    int batchSize = 2;

//    int numOutPlanes = 2;
//    int numInPlanes = 1;
//    int imageSize = 3;
//    int filterWidth = 3;
//    int padZeros = 0;
   
    LayerDimensions dim;
    dim.setNumFilters(2).setNumInputPlanes(1).setInputImageSize(3).setFilterSize(3)
        .setPadZeros(false).setBiased(false);

    float data[] = { 0, 0, 0,
                       -0.5f, 0.5f, 0,
                       0, 0, 0,

                        0, 0, 0,
                       0.5f, -0.5f, 0,
                       0, 0, 0

};

    float filter1[] = { 0, 0, 0,
                          0.300809f, -0.11011f, 0,
                         0, 0, 0,

                        0, 0, 0,
                          0.0570846f, 0.347077f, 0,
                         0,0,0

 };

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
//    float *results = new float[512];

    float *biases = 0;

//    CLWrapper *dataWrapper = cl->wrap( batchSize * 9, data );
//    CLWrapper *weightsWrapper = cl->wrap( numOutPlanes * 9, filter1 );
//    CLWrapper *resultsWrapper = cl->wrap( 512, results );
//    dataWrapper->copyToDevice();
//    weightsWrapper->copyToDevice();

//    CLKernel *convolve = cl->buildKernel( "../cl/propagate1.cl", "convolve_imagecubes_float2", "-D TANH" );
//    CLKernel *tanh = cl->buildKernel( "ClConvolve.cl", "byelement_tanh" );

//    for( int it = 0; it < 100; it ++ ) {

    Propagate *propagate = Propagate::instanceSpecific( 1, cl, dim, new TanhActivation() );
    float *results = propagate->propagate( batchSize, data, filter1, biases );

//        convolve->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( imageSize )->in( filterWidth )
//           ->in( padZeros );
//        convolve->input( dataWrapper );
//        convolve->input( weightsWrapper);
//        convolve->output( resultsWrapper );
//        int globalSize = batchSize * numOutPlanes * imageSize * imageSize;
//        int workgroupsize = cl->getMaxWorkgroupSize();
//        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
////        cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
//        convolve->run_1d( globalSize, workgroupsize );

//        resultsWrapper->copyToHost();

//        for( int i = 0; i < 20; i++ ) {
//            cout << "results[" << i << "]=" << results[i] << endl;
//        }
        EXPECT_FLOAT_NEAR( -0.202616f, results[0] );
        EXPECT_FLOAT_NEAR( 0.143989f, results[1] );
        EXPECT_FLOAT_NEAR( 0.202616f, results[2] );
        EXPECT_FLOAT_NEAR( -0.143989f, results[3] );
//    }
//    cout << "test2 ok" << endl;
    delete propagate;
//    delete convolve;
//    delete resultsWrapper;
//    delete weightsWrapper;
//    delete dataWrapper;
    delete[] results;
    delete cl;
}

TEST( testpropagate, test3 ) {
    int batchSize = 4;
    int numInPlanes = 2;
    int numOutPlanes = 2;
    int inImageSize = 1;
//    int outImageSize = 1;
    int filterSize = 1;
    int padZeros = 0;
    float data[] = {0.1f,0.2f,
                    0.3f,0.4f,
                    0.5f,0.6f,
                    0.7f,0.8f};
    float filter[] = {0.2f,0.3f,
                     0.5f,0.7f};

//    int outputImageSize = 0;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    Propagate *propagate = Propagate::instanceTest( cl, LayerDimensions( numInPlanes, inImageSize, numOutPlanes, filterSize,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( 
        batchSize, data, filter, 0 );        

    float expectedResults[] = {0.2f*0.1f+0.3f*0.2f,
                               0.5f*0.1f+0.7f*0.2f,

                               0.2f*0.3f+0.3f*0.4f,
                               0.5f*0.3f+0.7f*0.4f,

                                0.2f*0.5f+0.3f*0.6f,
                               0.5f*0.5f+0.7f*0.6f,
 
                              0.2f*0.7f+0.3f*0.8f,
                               0.5f*0.7f+0.7f*0.8f
  };
   for( int i = 0; i < 8; i++ ) {
//      cout << " checking result " << i << endl;
//        cout << "results[" << i << "]=" << results[i] << endl;
      assertEquals( expectedResults[i], results[i], 0.0001f);
   }
    delete propagate;
    delete cl;
}

void compareSpecific( bool debug, int N, int batchSize, LayerDimensions dim, ActivationFunction *fn, int instance0, int instance1 ) {
    cout << dim << endl;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int inputsSize = N * dim.inputCubeSize;
    int filtersSize = dim.filtersSize;
    int biasSize = dim.numFilters;
    int inputsAllocated = std::max( inputsSize, 10000 );
    int filtersAllocated = std::max( filtersSize, 10000 );
    int biasFiltersAllocated = std::max( biasSize, 10000 );
    float *inputs = new float[ inputsAllocated ];
    float *filters = new float[ filtersAllocated ];
    float *biasFilters = new float[ biasFiltersAllocated ];

    memset( inputs, 0, sizeof(float) * inputsAllocated );
    memset( filters, 0, sizeof(float) * filtersAllocated );
    memset( biasFilters, 0, sizeof(float) * biasFiltersAllocated );

//    inputs[0] = 2.0f;
//    inputs[1] = 4.0f;
    inputs[4] = 4.0f;
//    inputs[dim.inputB + 0] = 3.0f;
    inputs[dim.inputCubeSize + 0] = 3.0f;

//    filters[0] = 3.0f;
//    filters[1] = 5.0f;
    filters[4] = 5.0f;

    WeightRandomizer::randomize( inputs, inputsAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, filtersAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, biasFiltersAllocated, -0.1f, 0.1f );
    for( int i = 0; i < 8; i++ ) {
        if( debug ) cout << "i " << i << " input[i]=" << inputs[i] << " filters[i]=" << filters[i] << endl;
    }

    int resultsSize = N * dim.outputCubeSize;
    float *results1 = new float[ resultsSize ];
    float *results2 = new float[ resultsSize ];
    
    int numBatches = ( N + batchSize - 1 ) / batchSize;
    Propagate *p1 = Propagate::instanceSpecific( instance0, cl, dim, fn );
    Propagate *p2 = Propagate::instanceSpecific( instance1, cl, dim, fn );

//    float *resultstemps[2];
    for( int instance = 0; instance < 2; instance++ ) {
        Propagate *thisPropagate = 0;
        float *thisResults = 0;
        if( instance == 0 ) { 
            thisPropagate = p1;
            thisResults = results1;
        }
        if( instance == 1 ) {
            thisPropagate = p2;
            thisResults = results2;
        }
        for( int batch = 0; batch < numBatches; batch++ ) {
            int thisBatchSize = batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = N - batch * batchSize;
            }
            cout << "batch " << batch << " batchsize " << thisBatchSize << endl;
            float *resultstemp = new float[thisBatchSize * dim.outputCubeSize * sizeof(float)];
//            memset( resultstemp, 123, thisBatchSize * dim.outputCubeSize * sizeof(float) ); // so kernel
                // cant just reuse the work of previous propagate :-)
//            resultstemps[instance] = 
//            StatefulTimer::timeCheck("after memset");
            propagateWithWipe( thisPropagate, thisBatchSize, dim, inputs + batchSize * batch * dim.inputCubeSize, filters, biasFilters, resultstemp );
//            thisPropagate->propagate( thisBatchSize, inputs + batchSize * batch * dim.inputCubeSize, filters, biasFilters, resultstemp );
            memcpy( thisResults + batch * batchSize * dim.outputCubeSize, resultstemp, thisBatchSize * dim.outputCubeSize * sizeof(float) );
            delete[] resultstemp;
        }
        StatefulTimer::dump(true);
    }

    cout << dim << endl;
    bool same = true;
    int numDiff = 0;
    for( int i = 0; i < max( 20, resultsSize ); i++ ) {
        if( i < resultsSize ) {
            if( abs( results1[i] - results2[i] ) < 0.000001f || abs( results1[i] - results2[i] ) <= 0.001f * max( abs( results1[i] ), abs( results2[i] ) ) ) {
                if( i < 20 ) {
                    if( debug ) cout << "results[" << i << "]=" << results1[i] << " " << results2[i];
                    if( debug ) cout << " SAME";
                }
            } else {
                cout << "results[" << i << "]=" << results1[i] << " " << results2[i];
                cout << " DIFF";
                same = false;
                numDiff++;
            }
        } else {
             if( i < 20 ) {
                 if( debug ) cout << "     ";
             }
        }
        if( i < 20 ) {
            if( debug ) cout << "  || " << results2[100+i] ;
            if( debug ) cout << "  || " << results2[200+i] ;
            if( debug ) cout << "  || " << results2[300+i] ;
            if( debug ) cout << "  || " << results2[400+i] ;
            if( debug ) cout << "  || " << results2[500+i] ;
            if( debug ) cout << "  || " << results2[600+i] ;
            if( debug ) cout << "  || " << results2[700+i] << endl;
        }
        if( numDiff > 30 ) {
            cout << "..." << endl;
            break;
        }
    }
    EXPECT_EQ( true, same );
    delete[] results1;
    delete[] results2;
    delete p1;
    delete p2;
    delete cl;
    delete[] inputs;
    delete[] filters;
    delete[] biasFilters;
}

// first, compare the slow, but probably correct, cpu version, with propagate1
// propagate1 is slow-ish, but faster than cpu, and simple, so more likely to be correct
// then compare propagate1 with each other type
TEST( testpropagate, compare_0_1_biased_nopad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputImageSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( false ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    compareSpecific( false, N, batchSize, dim, fn, 0, 1 );
}

TEST( testpropagate, compare_0_1_biased_pad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputImageSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    compareSpecific( false, N, batchSize, dim, fn, 0, 1 );
}

TEST( testpropagate, compare_1_n_biased_nopad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputImageSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( false ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    for( int instance = 2; instance <= 7; instance++ ) {
        if( instance == 5 ) {
            continue; // propagatefc, cant use for inputimagesize != filtersize
        }
        cout << "instance: " << instance << endl;
        compareSpecific( false, N, batchSize, dim, fn, 1, instance );
    }
}

TEST( testpropagate, compare_1_n_biased_pad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputImageSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    for( int instance = 2; instance <= 7; instance++ ) {
        if( instance == 5 ) {
            continue; // propagatefc, cant use for inputimagesize != filtersize
        }
        cout << "instance: " << instance << endl;
        compareSpecific( false, N, batchSize, dim, fn, 1, instance );
    }
}

TEST( testpropagate, compare_1_5_biased_nopad ) { // only need to do nopad, since fc wont work with pad
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputImageSize(19).setNumFilters( 8 )
        .setFilterSize( 19 )
        .setPadZeros( false ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    compareSpecific( false, N, batchSize, dim, fn, 1, 5 );
}

TEST( testpropagate, compare_1_4_fcscenario ) { // only need to do nopad, since fc wont work with pad
    LayerDimensions dim;
    int batchSize = 4;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 10 ).setInputImageSize(24).setNumFilters( 10 )
        .setFilterSize( 24 )
        .setPadZeros( false ).setBiased( true );    
    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    compareSpecific( false, N, batchSize, dim, fn, 1, 4 );
}

//TEST( SLOW_testpropagate, comparespecific ) {
//    LayerDimensions dim;
//    dim.setInputPlanes( 2 ).setInputImageSize(5).setNumFilters( 1 ).setFilterSize( 5 )
//        .setPadZeros( true ).setBiased( false );    
//    compareSpecific( 1, dim, new LinearActivation(), 1, 3 );
//}

//TEST( SLOW_testpropagate, comparespecific_fc500unbiased ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 32 ).setInputImageSize(imageSize).setNumFilters( 500 ).setFilterSize( imageSize )
//        .setPadZeros( false ).setBiased( false );    
//    compareSpecific( 4, dim, new LinearActivation(), 1, 5 );
//}

//TEST( SLOW_testpropagate, comparespecific_fc500biased ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 32 ).setInputImageSize(imageSize).setNumFilters( 500 ).setFilterSize( imageSize )
//        .setPadZeros( false ).setBiased( true );    
//    compareSpecific( 4, dim, new LinearActivation(), 1, 5 );
//}

//TEST( SLOW_testpropagate, comparespecific_kgsgo_64c7 ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 64 ).setInputImageSize(imageSize).setNumFilters( 64 ).setFilterSize( 7 )
//        .setPadZeros( true ).setBiased( true );    
//    compareSpecific( 128, dim, new ReluActivation(), 1, 6 );
//}

TEST( SLOW_testpropagate, compare_args ) {
    LayerDimensions dim;
    int batchSize = 128;
//    int imageSize = 19;
//    int filterSize = 7;
//    int inputPlanes = 64;
//    int numFilters = 64;
    int instance0 = 1;
    int instance1 = 3;
    int N = 128;
    bool debug = false;
    string activationName = "tanh";
    dim.setInputPlanes( 64 ).setInputImageSize(19).setNumFilters( 64 )
        .setFilterSize( 7 )
        .setPadZeros( true ).setBiased( false );    

    TestArgsParser::arg( "n", &N );
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::arg( "debug", &debug );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::arg( "activation", &activationName );
    TestArgsParser::go();
    dim.deriveOthers();

    ActivationFunction *fn = ActivationFunction::fromName( activationName );
    compareSpecific( debug, N, batchSize, dim, fn, instance0, instance1 );
}

//TEST( SLOW_testpropagate, comparespecific_kgsgo_64c7mini ) {
//    LayerDimensions dim;
//    const int imageSize = 9;
//    dim.setInputPlanes( 4 ).setInputImageSize(imageSize).setNumFilters( 4 ).setFilterSize( 5 )
//        .setPadZeros( true ).setBiased( false );    
//    compareSpecific( 4, dim, new ReluActivation(), 1, 6 );
//}

TEST( testpropagate, softmax ) {
    NeuralNet *net = NeuralNet::maker()->imageSize(1)->planes(4)->instance();
    net->addLayer( SoftMaxMaker::instance() );
    net->setBatchSize( 1 );
    float *input = new float[net->getLayer(0)->getOutputPlanes()];
    input[0] = 0;
    input[1] = 1;
    input[2] = 3;
    input[3] = 2;
    net->propagate( input );
    float const*results = net->getResults();
    float sum = 0;
    for( int i = 0; i < net->getLayer(0)->getOutputPlanes(); i++ ) {
        cout << "results[" << i << "]=" << results[i] << endl;
        sum += results[i];
        EXPECT_LE( 0, results[i] );
        EXPECT_GE( 1, results[i] );
    }
    EXPECT_FLOAT_NEAR( 1.0f, sum );
    EXPECT_FLOAT_NEAR( (float)( exp(0.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[0] );
    EXPECT_FLOAT_NEAR( (float)( exp(1.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[1] );
    EXPECT_FLOAT_NEAR( (float)( exp(3.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[2] );
    EXPECT_FLOAT_NEAR( (float)( exp(2.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[3] );

    float *expected = new float[net->getLayer(0)->getOutputPlanes()];
    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[2] = 1;
    float loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[2]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[0] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[0]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[1] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[1]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[3] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[3]), loss );

    delete[] input;
    delete[] expected;
    delete net;
}

TEST( testpropagate, softmax_byplane ) {
    NeuralNet *net = NeuralNet::maker()->imageSize(2)->planes(1)->instance();
    net->addLayer( SoftMaxMaker::instance()->perPlane() );
    net->setBatchSize( 1 );
    int imageSizeSquared = net->getLayer(0)->getOutputImageSize() * net->getLayer(0)->getOutputImageSize();
    float *input = new float[imageSizeSquared];
    input[0] = 0;
    input[1] = 1;
    input[2] = 3;
    input[3] = 2;
    net->propagate( input );
    float const*results = net->getResults();
    float sum = 0;
    for( int i = 0; i < imageSizeSquared; i++ ) {
        cout << "results[" << i << "]=" << results[i] << endl;
        sum += results[i];
        EXPECT_LE( 0, results[i] );
        EXPECT_GE( 1, results[i] );
    }
    EXPECT_FLOAT_NEAR( 1.0f, sum );
    EXPECT_FLOAT_NEAR( (float)( exp(0.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[0] );
    EXPECT_FLOAT_NEAR( (float)( exp(1.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[1] );
    EXPECT_FLOAT_NEAR( (float)( exp(3.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[2] );
    EXPECT_FLOAT_NEAR( (float)( exp(2.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), results[3] );

    float *expected = new float[imageSizeSquared];
    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[2] = 1;
    float loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[2]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[0] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[0]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[1] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[1]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[3] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(results[3]), loss );

    delete[] input;
    delete[] expected;
    delete net;
}

void testPerf( int instance, int N, int batchSize, LayerDimensions dim, ActivationFunction *fn ) {
    cout << dim.buildOptionsString() << endl;  

    int inputsSize = batchSize * dim.inputCubeSize;
    int filtersSize = dim.filtersSize;
    int biasSize = dim.numFilters;
    int inputsAllocated = std::max( inputsSize, 10000 );
    int filtersAllocated = std::max( filtersSize, 10000 );
    int biasFiltersAllocated = std::max( biasSize, 10000 );
    float *inputs = new float[ inputsAllocated ];
    float *filters = new float[ filtersAllocated ];
    float *biasFilters = new float[ biasFiltersAllocated ];

    memset( inputs, 0, sizeof(float) * inputsAllocated );
    memset( filters, 0, sizeof(float) * filtersAllocated );
    memset( biasFilters, 0, sizeof(float) * biasFiltersAllocated );

    WeightRandomizer::randomize( inputs, inputsAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, filtersAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, biasFiltersAllocated, -0.1f, 0.1f );

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    Propagate *p1 = Propagate::instanceSpecific( instance, cl, dim, fn );
    for( int it = 0; it < (N + batchSize - 1 ) / batchSize; it++ ) {
        int thisBatchSize = it < N - 1 ? batchSize : N - batchSize * it;
        float *results1 = p1->propagate( thisBatchSize, inputs, filters, biasFilters );
        delete[] results1;
    }
    StatefulTimer::dump(true);

    delete p1;
    delete cl;
    delete[] inputs;
    delete[] filters;
    delete[] biasFilters;
}

TEST( SLOW_testpropagate, perf_kgsgo_fc500 ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(19).setNumFilters( 500 ).setFilterSize( 19 )
        .setPadZeros( false ).setBiased( true );  
    testPerf( -1, 128, batchSize, dim, new TanhActivation() );
}

TEST( SLOW_testpropagate, perf_mnist_firstconvlayer ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputImageSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim, new ReluActivation() );
}

TEST( SLOW_testpropagate, perf_mnist_intlayers_128ex ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim, new ReluActivation() );
}

TEST( SLOW_testpropagate, perf_mnist_intlayers_1024ex ) {
    int batchSize = 1024;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim, new ReluActivation() );
}

TEST( SLOW_testpropagate, perf_mnist_finallayer ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(28).setNumFilters( 10 ).setFilterSize( 28 )
        .setPadZeros( false ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim, new ReluActivation() );
}

TEST( SLOW_testpropagate, perf_kgsgo_64c7_args ) {
    int instance = 3;
    int batchSize = 128;
    int N = 1000;
    LayerDimensions dim;
    dim.setInputPlanes( 64 ).setInputImageSize(19).setNumFilters( 64 ).setFilterSize( 7 )
        .setPadZeros( true ).setBiased( true );  
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance", &instance );
    TestArgsParser::arg( "n", &N );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::go();
    testPerf( instance, N, batchSize, dim, new TanhActivation() );
}

