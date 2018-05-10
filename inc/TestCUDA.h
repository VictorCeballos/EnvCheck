#ifndef TESTCUDA_H
#define TESTCUDA_H

#include <cmath>
#include <memory>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <drvapi_error_string.h>


class TestCUDA
{

public:

    static void simple();
    static void execute(int argc, char **argv);
    static void execute2(int argc, char **argv);
};

#endif // TESTCUDA_H
