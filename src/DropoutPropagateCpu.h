// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DropoutPropagate.h"

#define VIRTUAL virtual
#define STATIC static

class DropoutPropagateCpu : public DropoutPropagate {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio );
    VIRTUAL void propagate( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper );
    VIRTUAL void propagate( int batchSize, unsigned char *masks, float *input, float *output );

    // [[[end]]]
};

