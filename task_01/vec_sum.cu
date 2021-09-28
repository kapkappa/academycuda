#include <assert.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

const double EPS = 1.e-15;

__global__ void vecAdd(double *d_a, double *d_b, double *d_c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
};

int main (int argc, char**argv) {
    assert(argc == 2);
    int n = atoi(argv[1]);
    //host vecs
    double *h_a, *h_b, *h_c;

    size_t bytes = n * sizeof(double);

    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

double t1 = timer();

    for (int i = 0; i < n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

double t2 = timer();
std::cout << "cpu time is: " << t2-t1 << std::endl;

    //dev imput vecs
    double *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    //num of threads in each thread block
    blockSize = 1024;
    //num of thread blocks in grid
    gridSize = (n-1)/blockSize + 1;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

double t3 = timer();

cudaEventRecord(start);

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

double t4 = timer();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "(Cuda runtime) Gpu time is: " << t4-t3 << std::endl;
std::cout << "(Timer) Gpu time is: " << ms << std::endl;

    for (int i = 0; i < n; i++) {
        if (abs(h_c[i]-h_a[i]-h_b[i]) >= EPS) {
            std::cout << "CHECK FAILED!" << std::endl;
            std::cout << "DIFFER: " << abs(h_c[i] - h_a[i] - h_b[i]) << std::endl;
            break;
        }
    }

    std::cout << "Check completed!" << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
