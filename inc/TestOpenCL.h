#ifndef TESTOPENCL_H
#define TESTOPENCL_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <ctime>


class TestOpenCL
{

public:

    static void clPrintDevInfo(cl_device_id device);
    static void execute(int argc, char **argv);
};

#endif // TESTOPENCL_H
