#include <assert.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

const double EPS = 1.e-15;
const int CHUNK = 8;

__global__ void vecAdd(double *d_a, double *d_b, double *d_c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (N / CHUNK)) {
//        int width = gridDim.x * blockDim.x;
        int width = N / CHUNK;
        for (int i = 0;  i < CHUNK; i++)
            if (idx % 2 == 0)
                d_c[idx + i * width] = d_a[idx + i * width] + d_b[idx + i * width];
            else
                d_c[idx + i * width] = d_a[idx + i * width] - d_b[idx + i * width];
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
    assert(n % CHUNK == 0);
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

    int blockSize = 1024;
    int gridSize = ((n-1)/CHUNK)/blockSize + 1;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Gpu time is: " << ms << std::endl;

    for (int i = 0; i < n; i++) {
        if (abs(h_c[i]-h_a[i]-h_b[i]) >= EPS) {
            std::cout << "CHECK FAILED!\t";
            std::cout << "DIFFER: " << abs(h_c[i] - h_a[i] - h_b[i]) << std::endl;
            break;
        } // else std::cout << "CHECK PASSED" << std::endl;
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
