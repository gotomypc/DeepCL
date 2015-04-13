// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NorbLoader.h"
#include "FileHelper.h"
#include "Kgsv2Loader.h"
#include "StatefulTimer.h"
#include "MnistLoader.h"

#include "GenericLoader.h"

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void GenericLoader::getDimensions( std::string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) {
    char *headerBytes = FileHelper::readBinaryChunk( trainFilepath, 0, 1024 );
    char type[1025];
    strncpy( type, headerBytes, 4 );
    type[4] = 0;
    unsigned int *headerInts = reinterpret_cast< unsigned int *>( headerBytes );
    if( string(type) == "mlv2" ) {
//        cout << "Loading as a Kgsv2 file" << endl;
        Kgsv2Loader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else if( headerInts[0] == 0x1e3d4c55 ) {
//        cout << "Loading as a Norb mat file" << endl;
        NorbLoader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else if( headerInts[0] == 0x03080000 ) {
        MnistLoader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + trainFilepath + " not recognised" );
    }
}

STATIC void GenericLoader::load( std::string imagesFilepath, unsigned char *images, int *labels ) {
    load( imagesFilepath, images, labels, 0, 0 );
}

STATIC void GenericLoader::load( std::string imagesFilepath, unsigned char *images, int startN, int numExamples ) {
    load( imagesFilepath, images, 0, startN, numExamples );
}

STATIC void GenericLoader::load( std::string imagesFilepath, unsigned char *images, int *labels, int startN, int numExamples ) {
    StatefulTimer::timeCheck("GenericLoader::load start");
    char *headerBytes = FileHelper::readBinaryChunk( imagesFilepath, 0, 1024 );
    char type[1025];
    strncpy( type, headerBytes, 4 );
    type[4] = 0;
    unsigned int *headerInts = reinterpret_cast< unsigned int *>( headerBytes );
    if( string(type) == "mlv2" ) {
//        cout << "Loading as a Kgsv2 file" << endl;
        Kgsv2Loader::load( imagesFilepath, images, labels, startN, numExamples );
    } else if( headerInts[0] == 0x1e3d4c55 ) {
//        cout << "Loading as a Norb mat file" << endl;
        NorbLoader::load( imagesFilepath, images, labels, startN, numExamples );
    } else if( headerInts[0] == 0x03080000 ) {
        MnistLoader::load( imagesFilepath, images, labels, startN, numExamples );
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + imagesFilepath + " not recognised" );
    }
    StatefulTimer::timeCheck("GenericLoader::load end");
}


