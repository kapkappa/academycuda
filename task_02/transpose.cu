#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <assert.h>
#include <sys/time.h>

const int B_SIZE = 1024;

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

__global__ void transpose(double *M, double *M_t, int n) {

    int Idx = blockIdx.x * B_SIZE + threadIdx.x;

    if(Idx < n*n) {
        int x = Idx % n;
        int y = Idx / n;
        int T_index = x * n + y;
        M_t[T_index] = M[Idx];
    }
}

void print(double *M, int n) {
    std::cout << "Print matrix: \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << M[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    int n = atoi(argv[1]);
    size_t size = n * n * sizeof(double);
    double *M = (double*)malloc(size);

    //init
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i * n + j] = i / j + j % i;
        }
    }

    double *M_t = (double*)malloc(size);

    double t1 = timer();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M_t[j * n + i] = M[i * n + j];
        }
    }

    double t2 = timer();
    std::cout << "Cpu time: " << t2-t1 << std::endl;

//    print(M, n);
//    print(M_t, n);

    dim3 block(B_SIZE);
    dim3 grid((n * n - 1) / B_SIZE + 1);

    double *M_dev, *M_t_dev;
    cudaMalloc(&M_dev, size);
    cudaMalloc(&M_t_dev, size);

    cudaMemcpy(M_dev, M, size, cudaMemcpyHostToDevice);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

    transpose<<<grid, block>>>(M_dev, M_t_dev, n);

cudaDeviceSynchronize();
cudaEventRecord(stop);

    cudaMemcpy(M_t, M_t_dev, size, cudaMemcpyDeviceToHost);

cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Gpu time is: " << ms << std::endl;

//    print(M_t, n);

    bool check = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            check *= (M[i*n+j] == M_t[j*n+i]);
        }
    }
    if (check) std::cout << "CHECK PASSED!" << std::endl;
    else std::cout << "CHECK FAILED!" << std::endl;

    free(M);
    free(M_t);

    cudaFree(M_dev);
    cudaFree(M_t_dev);

    return 0;
}
