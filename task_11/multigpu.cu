// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#define BLOCKSIZE 1024

const float EPS = 1.e-06;

inline
cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

void unified_sample (int size = 1048576) {
    int n = size;
    int nBytes = n * sizeof(float);

    float *a, *b, *c;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(BLOCKSIZE);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);
    cudaMallocManaged(&c, nBytes);

    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    cudaEventRecord(start);

    vectorAddGPU<<<grid, block>>>(a, b, c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("UNI: Memalloc(unified memory) + Kernel time is: %f\n", ms);

    cudaDeviceSynchronize();

    printf("NO CHECK!\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void pinned_sample (int size = 1048576) {
    int n = size;
    size_t nBytes = n * sizeof(float);
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
//    float errNorm, refNorm, ref, diff;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(BLOCKSIZE);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    cudaMallocHost(&a, nBytes);
    cudaMallocHost(&b, nBytes);
    cudaMallocHost(&c, nBytes);
    cudaMalloc(&d_a, nBytes);
    cudaMalloc(&d_b, nBytes);
    cudaMalloc(&d_c, nBytes);

    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    cudaEventRecord(start);

    cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice);
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, nBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("PIN: Memcpy + Kernel time is: %f\n", ms);

    cudaDeviceSynchronize();

    //CHECK
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - a[i] - b[i]) > EPS) {
            printf("CHECK FAILED!!!\n");
            printf("Differ: %f\n", c[i]-a[i]-b[i]);
            break;
        }
    }
    printf("CHECK PASSED!\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
}

void usual_sample (int size = 1048576) {
    int n = size;
    int nBytes = n*sizeof(float);

    float *a, *b;  // host data
    float *c;  // results

    a = (float*)malloc(nBytes);
    b = (float*)malloc(nBytes);
    c = (float*)malloc(nBytes);

    float *a_d,*b_d,*c_d;

    dim3 block(BLOCKSIZE);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    for(int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    printf("Allocating device memory on host..\n");

    cudaEvent_t start, stop, malloc_start, malloc_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
//    cudaEventCreate(&malloc_start);
//    cudaEventCreate(&malloc_stop);

    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));

    cudaEventRecord(start);

    cudaMemcpy(a_d, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice);

    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    cudaMemcpy(c, c_d, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("USUAL Memcpy + Kernel: %f ms\n", milliseconds);

    cudaDeviceSynchronize();

    //CHECK
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - a[i] - b[i]) > EPS) {
            printf("CHECK FAILED!!!\n");
            printf("Differ: %f\n", c[i]-a[i]-b[i]);
            break;
        }
    }
    printf("CHECK PASSED!\n");

    free(a);
    free(b);
    free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void multigpu (int size) {
    int n = size;
    int nBytes = n * sizeof(float);

    dim3 block(BLOCKSIZE);

    float *a, *b, *c;
    cudaHostAlloc((void**)&a, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, nBytes, cudaHostAllocDefault);

    for (int i = 0; i < n; i++) {
//        a[i] = rand() / (float)RAND_MAX;
//        b[i] = rand() / (float)RAND_MAX;
        a[i] = i;
        b[i] = 1;
        c[i] = 0;
    }

    int deviceCnt;
    cudaGetDeviceCount(&deviceCnt);
      deviceCnt = 2;
    printf("Devices: %d", deviceCnt);

    int bytes_per_device = nBytes / deviceCnt + 1;

//    float (*a_d)[deviceCnt], (*b_d)[deviceCnt], (*c_d)[deviceCnt];
    float **a_d, **b_d, **c_d;
    a_d = (float**)malloc(sizeof(float*) * deviceCnt);
    b_d = (float**)malloc(sizeof(float*) * deviceCnt);
    c_d = (float**)malloc(sizeof(float*) * deviceCnt);


    int n_per_device = (n-1)/deviceCnt + 1;
    dim3 grid((n_per_device-1)/BLOCKSIZE + 1);

    cudaEvent_t start[deviceCnt], stop[deviceCnt];
    for (int i = 0; i < deviceCnt; i++) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }

    printf("\nParallel starts here\n\n");

    for (int i = 0; i < deviceCnt; i++) {
        cudaSetDevice(i);
        int check;
        cudaGetDevice(&check);
        assert(i == check);
        cudaMalloc((void**)&(a_d[i]), bytes_per_device);
        cudaMalloc((void**)&(b_d[i]), bytes_per_device);
        cudaMalloc((void**)&(c_d[i]), bytes_per_device);
    }

    for (int i = 0; i < deviceCnt; i++) {
        cudaSetDevice(i);
//        int check;
//        cudaGetDevice(&check);
//        assert(i == check);
        printf("i am on device %d\nposition is %d\n", i, n_per_device*i);
        cudaEventRecord(start[i]);
        cudaMemcpyAsync(a_d[i], a + n_per_device * i, bytes_per_device, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(b_d[i], b + n_per_device * i, bytes_per_device, cudaMemcpyHostToDevice);

        vectorAddGPU<<<grid, block>>>(a_d[i], b_d[i], c_d[i], n);

        cudaMemcpyAsync(c + n_per_device * i, c_d[i], bytes_per_device, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop[i]);
    }

    for (int dev = 0; dev < deviceCnt; dev++) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < deviceCnt; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, start[i], stop[i]);
        printf("Elapsed time on device %d is %f\n", i, ms);
    }

    for (int i = 0; i < deviceCnt; i++) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }
/*
    for (int i = 0; i < n; i++)
        printf("%f ", c[i]);
    printf("\n\n");
*/
    //CHECK
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - a[i] - b[i]) > EPS) {
            printf("CHECK FAILED!!!\n");
            printf("Differ: %f\n", c[i]-a[i]-b[i]);
            break;
        }
    }
    printf("CHECK PASSED!\n");

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    for (int i = 0; i < deviceCnt; i++) {
        cudaFree(a_d[i]);
        cudaFree(b_d[i]);
        cudaFree(c_d[i]);
    }
}

void thread_gpu (int size) {
    int n = size;
    int nBytes = n * sizeof(float);

    float *a, *b, *c;
    float *a_d, *b_d, *c_d;

    int deviceCnt;
    cudaGetDeviceCount(&deviceCnt);
    printf("Devices: %d\n", deviceCnt);

    dim3 block(BLOCKSIZE);
    dim3 grid((unsigned int)ceil(n/(float)block.x) / deviceCnt);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&start);

    cudaHostAlloc(&a, nBytes, 0);
    cudaHostAlloc(&b, nBytes, 0);
    cudaHostAlloc(&c, nBytes, 0);

    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    int n_per_device = n / deviceCnt + 1;
    int bytes_per_device = nBytes / deviceCnt + 1;
#pragma omp parallel num_threads(deviceCnt)
    {
        int device = omp_get_thread_num();
        cudaSetDevice(device);
        cudaMalloc(&a_d, bytes_per_device);
        cudaMalloc(&b_d, bytes_per_device);
        cudaMalloc(&c_d, bytes_per_device);

        cudaEventRecord(start);
        cudaMemcpy(a_d, a + device * n_per_device, bytes_per_device, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, c + device * n_per_device, bytes_per_device, cudaMemcpyHostToDevice);

        vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n_per_device);
        cudaMemcpy(c + device * n_per_device, c_d, bytes_per_device, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Elapsed time: %f\n", ms);
        cudaFree(a_d);
        cudaFree(b_d);
        cudaFree(c_d);
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}

int main(int argc, char **argv) {
    assert(argc==3);
    switch (atoi(argv[2])) {
    case 0:
        usual_sample(atoi(argv[1]));
        break;
    case 1:
        pinned_sample(atoi(argv[1]));
        break;
    case 2:
        unified_sample(atoi(argv[1]));
        break;
    case 3:
        multigpu(atoi(argv[1]));
        break;
    default:
        break;
    }
    return 0;
}
