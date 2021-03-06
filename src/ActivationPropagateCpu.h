// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActivationPropagate.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationPropagateCpu : public ActivationPropagate {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    VIRTUAL void propagate( int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper );
    VIRTUAL void propagate( int batchSize, float *input, float *output );

    // [[[end]]]
};

