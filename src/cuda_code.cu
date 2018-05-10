__global__ void
cudaAddVectorsKernel(float *a, float *b, float *c, int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < size) {
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
    }
}

extern "C"
void cudaAddVectors(const float* a, const float* b, float* c, int size) {
    // For now, suppose a and b were created before calling this function

    // dev_a, dev_b (for inputs) and dev_c (for outputs) will be
    // arrays on the GPU.

    float * dev_a;
    float * dev_b;

    float * dev_c;

    // Allocate memory on the GPU for our inputs:
    cudaMalloc((void **) &dev_a, size*sizeof(float));
    cudaMemcpy(dev_a, a, size*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_b, size*sizeof(float)); // and dev_b
    cudaMemcpy(dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for our outputs:
    cudaMalloc((void **) &dev_c, size*sizeof(float));

    // At lowest, should be 32
    // Limit of 512 (Tesla), 1024 (newer)
    const unsigned int threadsPerBlock = 512;

    // For performance reasons, this shouldn't be too high
    const unsigned int maxBlocks = 100;

    // How many block we'll end up needing
    const unsigned int blocks = min(maxBlocks,
        (int) ceil(size/float(threadsPerBlock)));

    // Call the kernel!
    cudaAddVectorsKernel<<<blocks, threadsPerBlock>>>
        (dev_a, dev_b, dev_c, size);

    // Copy output from device to host (assume here that host memory
    // for the output has been calculated)

    cudaMemcpy(c, dev_c, size*sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

}

__global__
void cudaWaveKernel(float *d_data, size_t size, size_t timestep, const float courantSquared) {

    // Get unique thread id
    unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Pointers to data
    const int oldData     = globalID + ( (timestep + 0) % 3 ) * size;
    const int currentData = globalID + ( (timestep + 1) % 3 ) * size;
    const int newData     = globalID + ( (timestep + 2) % 3 ) * size;

    if (globalID < size - 1) {

        // Update data
        d_data[newData] = 2 * d_data[currentData] - d_data[oldData]
         + courantSquared * (d_data[currentData+1] - 2*d_data[currentData] + d_data[currentData-1] );

    }

}

extern "C"
void cudaWave( const unsigned int blocks, const unsigned int threadsPerBlock,
        float *d_data, size_t numberOfNodes, size_t timestepIndex, const float courantSquared ) {

    cudaWaveKernel<<< blocks, threadsPerBlock >>>
            (d_data, numberOfNodes, timestepIndex, courantSquared);
}
