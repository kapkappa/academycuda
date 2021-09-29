#include <iostream>
#include <assert.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

const int SIZE = 32;

__global__ void transpose(int *V, int n) {
//    __shared__ int array[SIZE];

    int Idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (Idx <=  n/2) {
        int tmp = V[Idx];
        V[Idx] = V[n-Idx-1];
        V[n-Idx-1] = tmp;
    }
}

int main(int argc, char **argv) {
    assert(argc==2);
    int n = atoi(argv[1]);

    size_t size = n * sizeof(int);
    int *V = (int*)malloc(size);

    for (int i = 0; i < n; i++) {
        V[i] = i;
    }

    int block = 1024;
    int grid = (n / 2 - 1) / block + 1;

    int *V_t;
    cudaMalloc(&V_t, size);
    cudaMemcpy(V_t, V, size, cudaMemcpyHostToDevice);

    transpose<<<grid, block>>>(V_t, n);

    cudaDeviceSynchronize();

    cudaMemcpy(V, V_t, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << V[i] << " ";
    }
    std::cout << std::endl;

    free(V);
    cudaFree(V_t);

    return 0;
}
