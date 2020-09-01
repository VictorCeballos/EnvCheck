#include <iostream>
#include "inc/TestOpenCL.h"
#include "inc/TestOpenGL.h"
#include "inc/TestCUDA.h"

using namespace std;

int main(int argc, char **argv)
{
    cout << "Hello World!" << endl;
    TestOpenCL::execute(argc, argv);
    TestCUDA::simple();
    TestCUDA::deviceQuery(argc, argv);
    TestCUDA::deviceQueryDriver(argc, argv);
    TestOpenGL::execute(argc, argv);
    return 0;
}

