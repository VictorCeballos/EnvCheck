#include "inc/TestCUDA.h"

// This is the 'elder trick of the...' - Tell the compiler this function is defined in other place
extern "C"
void cudaAddVectors(const float* a, const float* b, float* c, int size);

extern "C"
void cudaWave( const unsigned int blocks, const unsigned int threadsPerBlock,
               float *d_data, size_t numberOfNodes, size_t timestepIndex, const float courantSquared);

void TestCUDA::simple()
{
    float a[5] = {1,2,3,4,5};
    float b[5] = {6,7,8,9,0};
    float c[5];
    int size = 5;

    cudaAddVectors(a, b, c, size);

    int i;
    for (i=0;i<5;i++)
        printf("%d %f\n", i, c[i]);

    printf("\n");

    const int numberOfIntervals = 14;
    const size_t numberOfNodes = numberOfIntervals + 1;
    const float courant = 1.0;
    const float courantSquared = courant * courant;
    const float omega0 = 10;
    const float omega1 = 100;
    const float dx = 1./numberOfIntervals;
    const float dt = courant * dx;

    const unsigned int blocks = 3;
    const unsigned int threadsPerBlock = 5;

    // Space on the CPU to copy file data back from GPU
    float *file_output = new float[numberOfNodes];

    float *dev_pos;
    cudaMalloc((void **) &dev_pos, 3*numberOfNodes*sizeof(float));
    cudaMemset(dev_pos, 0, 3*numberOfNodes*sizeof(float));

    float initial[45] = {7, 2, 5, 3, 1, 2, 1, 5, 4, 6, 7, 8, 10, 11, 5,
                         1, 3, 7, 8, 5, 6, 1, 2, 3, 3, 5, 7,  9, 13, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0 };

    cudaMemcpy(dev_pos, initial, 3*numberOfNodes*sizeof(float),
               cudaMemcpyHostToDevice);

    size_t timestepIndex = 0;

    // TODO: Call a kernel to solve the problem (you'll need to make
    // the kernel in the .cu file)
    cudaWave(blocks, threadsPerBlock, dev_pos, numberOfNodes, timestepIndex, courantSquared);

    // Left boundary condition on the CPU - a sum of sine waves
    const float t = timestepIndex * dt;
    float left_boundary_value;
    if (omega0 * t < 2 * M_PI) {
        left_boundary_value = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
    } else {
        left_boundary_value = 0;
    }

    // TODO: Apply left and right boundary conditions on the GPU.
    // The right boundary condition will be 0 at the last position
    // for all times t
    int newData = ( (timestepIndex + 2) % 3 ) * numberOfNodes;

    left_boundary_value = 67;
    float k = 77;

    cudaMemcpy( &dev_pos[newData], &left_boundary_value,
                sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy( &dev_pos[newData + numberOfNodes - 1], &k,
            sizeof(float), cudaMemcpyHostToDevice);

    // TODO: Copy data from GPU back to the CPU in file_output
    cudaMemcpy(file_output, &dev_pos[newData],
               numberOfNodes*sizeof(float), cudaMemcpyDeviceToHost);

    for (i=0;i<15;i++)
        printf("%d %f\n", i, file_output[i]);

    printf("\n");
}

void TestCUDA::execute(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
                deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
                deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
                deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2)
    {
        cudaDeviceProp prop[64];
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i=0; i < deviceCount; i++)
        {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                    // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
                    && prop[i].tccDriver
        #endif
                    )
            {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2)
        {
            for (int i = 0; i < gpu_p2p_count; i++)
            {
                for (int j = 0; j < gpu_p2p_count; j++)
                {
                    if (gpuid[i] == gpuid[j])
                    {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, gpuid[i],
                            prop[gpuid[j]].name, gpuid[j] ,
                            can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
        sprintf(cTemp, ", Device%d = ", dev);
#endif
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += cTemp;
        sProfileString += deviceProp.name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");
}

void TestCUDA::execute2(int argc, char **argv)
{
    CUdevice dev;
    int major = 0, minor = 0;
    int deviceCount = 0;
    char deviceName[256];

    printf("%s Starting...\n\n", argv[0]);

    // note your project will need to link with cuda.lib files on windows
    printf("CUDA Device Query (Driver API) statically linked version \n");

    CUresult error_id = cuInit(0);

    if (error_id != CUDA_SUCCESS)
    {
        printf("cuInit(0) returned %d\n-> %s\n", error_id, getCudaDrvErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    error_id = cuDeviceGetCount(&deviceCount);

    if (error_id != CUDA_SUCCESS)
    {
        printf("cuDeviceGetCount returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (dev = 0; dev < deviceCount; ++dev)
    {
        error_id =  cuDeviceComputeCapability(&major, &minor, dev);

        if (error_id != CUDA_SUCCESS)
        {
            printf("cuDeviceComputeCapability returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        error_id = cuDeviceGetName(deviceName, 256, dev);

        if (error_id != CUDA_SUCCESS)
        {
            printf("cuDeviceGetName returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        printf("\nDevice %d: \"%s\"\n", dev, deviceName);

        int driverVersion = 0;
        cuDriverGetVersion(&driverVersion);
        printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", major, minor);

        size_t totalGlobalMem;
        error_id = cuDeviceTotalMem(&totalGlobalMem, dev);

        if (error_id != CUDA_SUCCESS)
        {
            printf("cuDeviceTotalMem returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)totalGlobalMem/1048576.0f, (unsigned long long) totalGlobalMem);
        printf("%s", msg);

        int multiProcessorCount;
        getCudaAttribute<int>(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               multiProcessorCount, _ConvertSMVer2CoresDRV(major, minor),
               _ConvertSMVer2CoresDRV(major, minor) * multiProcessorCount);

        int clockRate;
        getCudaAttribute<int>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", clockRate * 1e-3f, clockRate * 1e-6f);
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

        int maxTex1D, maxTex2D[2], maxTex3D[3];
        getCudaAttribute<int>(&maxTex1D, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, dev);
        getCudaAttribute<int>(&maxTex2D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, dev);
        getCudaAttribute<int>(&maxTex2D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, dev);
        getCudaAttribute<int>(&maxTex3D[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev);
        getCudaAttribute<int>(&maxTex3D[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev);
        getCudaAttribute<int>(&maxTex3D[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev);
        printf("  Max Texture Dimension Sizes                    1D=(%d) 2D=(%d, %d) 3D=(%d, %d, %d)\n",
               maxTex1D, maxTex2D[0], maxTex2D[1], maxTex3D[0], maxTex3D[1], maxTex3D[2]);

        int  maxTex1DLayered[2];
        getCudaAttribute<int>(&maxTex1DLayered[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, dev);
        getCudaAttribute<int>(&maxTex1DLayered[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, dev);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               maxTex1DLayered[0], maxTex1DLayered[1]);

        int  maxTex2DLayered[3];
        getCudaAttribute<int>(&maxTex2DLayered[0], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, dev);
        getCudaAttribute<int>(&maxTex2DLayered[1], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, dev);
        getCudaAttribute<int>(&maxTex2DLayered[2], CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, dev);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               maxTex2DLayered[0], maxTex2DLayered[1], maxTex2DLayered[2]);

        int totalConstantMemory;
        getCudaAttribute<int>(&totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
        printf("  Total amount of constant memory:               %u bytes\n", totalConstantMemory);
        int sharedMemPerBlock;
        getCudaAttribute<int>(&sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
        printf("  Total amount of shared memory per block:       %u bytes\n", sharedMemPerBlock);
        int regsPerBlock;
        getCudaAttribute<int>(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
        printf("  Total number of registers available per block: %d\n", regsPerBlock);
        int warpSize;
        getCudaAttribute<int>(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
        printf("  Warp size:                                     %d\n", warpSize);
        int maxThreadsPerMultiProcessor;
        getCudaAttribute<int>(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
        printf("  Maximum number of threads per multiprocessor:  %d\n", maxThreadsPerMultiProcessor);
        int maxThreadsPerBlock;
        getCudaAttribute<int>(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
        printf("  Maximum number of threads per block:           %d\n", maxThreadsPerBlock);

        int blockDim[3];
        getCudaAttribute<int>(&blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
        getCudaAttribute<int>(&blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
        getCudaAttribute<int>(&blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", blockDim[0], blockDim[1], blockDim[2]);
        int gridDim[3];
        getCudaAttribute<int>(&gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
        getCudaAttribute<int>(&gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
        getCudaAttribute<int>(&gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
        printf("  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n", gridDim[0], gridDim[1], gridDim[2]);

        int textureAlign;
        getCudaAttribute<int>(&textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
        printf("  Texture alignment:                             %u bytes\n", textureAlign);

        int memPitch;
        getCudaAttribute<int>(&memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, dev);
        printf("  Maximum memory pitch:                          %u bytes\n", memPitch);

        int gpuOverlap;
        getCudaAttribute<int>(&gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);

        int asyncEngineCount;
        getCudaAttribute<int>(&asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (gpuOverlap ? "Yes" : "No"), asyncEngineCount);

        int kernelExecTimeoutEnabled;
        getCudaAttribute<int>(&kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
        printf("  Run time limit on kernels:                     %s\n", kernelExecTimeoutEnabled ? "Yes" : "No");
        int integrated;
        getCudaAttribute<int>(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
        printf("  Integrated GPU sharing Host Memory:            %s\n", integrated ? "Yes" : "No");
        int canMapHostMemory;
        getCudaAttribute<int>(&canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
        printf("  Support host page-locked memory mapping:       %s\n", canMapHostMemory ? "Yes" : "No");

        int concurrentKernels;
        getCudaAttribute<int>(&concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
        printf("  Concurrent kernel execution:                   %s\n", concurrentKernels ? "Yes" : "No");

        int surfaceAlignment;
        getCudaAttribute<int>(&surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev);
        printf("  Alignment requirement for Surfaces:            %s\n", surfaceAlignment ? "Yes" : "No");

        int eccEnabled;
        getCudaAttribute<int>(&eccEnabled,  CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev);
        printf("  Device has ECC support:                        %s\n", eccEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        int tccDriver ;
        getCudaAttribute<int>(&tccDriver ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev);
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif

        int unifiedAddressing;
        getCudaAttribute<int>(&unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
        printf("  Device supports Unified Addressing (UVA):      %s\n", unifiedAddressing ? "Yes" : "No");

        int pciDomainID, pciBusID, pciDeviceID;
        getCudaAttribute<int>(&pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
        getCudaAttribute<int>(&pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
        getCudaAttribute<int>(&pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", pciDomainID, pciBusID, pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };

        int computeMode;
        getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[computeMode]);
    }


    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2)
    {
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;
        int tccDriver = 0;

        for (int i=0; i < deviceCount; i++)
        {
            checkCudaErrors(cuDeviceComputeCapability(&major, &minor, i));
            getCudaAttribute<int>(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i);

            // Only boards based on Fermi or later can support P2P
            if ((major >= 2)
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                    // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
                    && tccDriver
        #endif
                    )
            {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;
        char deviceName0[256], deviceName1[256];

        if (gpu_p2p_count >= 2)
        {
            for (int i = 0; i < gpu_p2p_count; i++)
            {
                for (int j = 0; j < gpu_p2p_count; j++)
                {
                    if (gpuid[i] == gpuid[j])
                    {
                        continue;
                    }
                    checkCudaErrors(cuDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    checkCudaErrors(cuDeviceGetName(deviceName0, 256, gpuid[i]));
                    checkCudaErrors(cuDeviceGetName(deviceName1, 256, gpuid[j]));
                    printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", deviceName0, gpuid[i],
                           deviceName1, gpuid[j] ,
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    printf("Result = PASS\n");
}
