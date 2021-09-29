#include <iostream>
#include <assert.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

const int SIZE = 1024;

__global__ void transpose(int *V, int n) {
    int Idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (Idx <=  n/2) {
        int tmp = V[Idx];
        V[Idx] = V[n-Idx-1];
        V[n-Idx-1] = tmp;
    }
}

__global__ void static_trans(int *Vec, int N) {
    __shared__ int array[SIZE];

    int Idx = threadIdx.x; //we have only one block - condition
    array[Idx] = Vec[Idx];

    __syncthreads();

    Vec[Idx] = array[N-Idx-1];
}

__global__ void dynamic_trans(int *Vec, int N) {
    extern __shared__ int array[];

    int Idx = threadIdx.x;
    array[Idx] = Vec[Idx];

    __syncthreads();

    Vec[Idx] = array[N-Idx-1];
}


int main(int argc, char **argv) {
    assert(argc==2);
    int n = atoi(argv[1]);

    size_t size = n * sizeof(int);
    int *V = (int*)malloc(size);
    int *V_t = (int*)malloc(size);

    for (int i = 0; i < n; i++) {
        V[i] = i;
    }

    int block = 1024;
    int grid = 1;
//    int grid = (n / 2 - 1) / block + 1;

    int *V_dev;
    cudaMalloc(&V_dev, size);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

    cudaMemcpy(V_dev, V, size, cudaMemcpyHostToDevice);
    transpose<<<grid, block>>>(V_dev, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

    cudaMemcpy(V_t, V_dev, size, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Common time is: " << ms << std::endl;

    for (int i = 0; i < n; i++) {
        std::cout << V_t[i] << " ";
    }
    std::cout << std::endl;

/////////////////////////////////////////////////
//STATIC SHARED

cudaEventRecord(start);

    cudaMemcpy(V_dev, V, size, cudaMemcpyHostToDevice);
    static_trans<<<grid, block, n>>>(V_dev, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

    cudaMemcpy(V_t, V_dev, size, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Static sh m time is: " << ms << std::endl;
   for (int i = 0; i < n; i++) {
        std::cout << V_t[i] << " ";
    }
    std::cout << std::endl;

/////////////////////////////////////////////////
//DYNAMIC SHARED

cudaEventRecord(start);

    cudaMemcpy(V_dev, V, size, cudaMemcpyHostToDevice);
    dynamic_trans<<<grid, block, n>>>(V_dev, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

    cudaMemcpy(V_t, V_dev, size, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Dynamic sh m time is: " << ms << std::endl;
   for (int i = 0; i < n; i++) {
        std::cout << V_t[i] << " ";
    }
    std::cout << std::endl;


    free(V);
    cudaFree(V_t);

    return 0;
}
