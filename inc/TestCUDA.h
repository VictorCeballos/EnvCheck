#ifndef TESTCUDA_H
#define TESTCUDA_H

#include <cmath>
#include <memory>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


class TestCUDA
{

public:

    static void simple();
    static void deviceQuery(int argc, char **argv);
    static void deviceQueryDriver(int argc, char **argv);
};

#endif // TESTCUDA_H
