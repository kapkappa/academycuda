// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 256

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

    cudaEvent_t uniStart, uniStop;
    cudaEventCreate(&uniStart);
    cudaEventCreate(&uniStop);

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

    cudaEventRecord(uniStart);

    vectorAddGPU<<<grid, block>>>(a, b, c, n);

    cudaEventRecord(uniStop);
    cudaEventSynchronize(uniStop);
    float ms = 0;
    cudaEventElapsedTime(&ms, uniStart, uniStop);
    printf("UNI: Memalloc(unified memory) + Kernel time is: %f\n", ms);

    cudaDeviceSynchronize();

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

    cudaEvent_t pinStart, pinStop;
    cudaEventCreate(&pinStart);
    cudaEventCreate(&pinStop);

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

    cudaEventRecord(pinStart);

    cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice);
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, nBytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(pinStop);
    cudaThreadSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, pinStart, pinStop);
    printf("PIN: Memcpy + Kernel time is: %f\n", ms);

    cudaDeviceSynchronize();

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
    cudaThreadSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("USUAL Memcpy + Kernel: %f ms\n", milliseconds);

    cudaDeviceSynchronize();

    free(a);
    free(b);
    free(c);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void multigpu (int size) {
    printf("DEBUG0\n");
    int n = size;
    int nBytes = n * sizeof(float);

    dim3 block(BLOCKSIZE);

    printf("DEBUG1\n");

    float *a, *b, *c;
    cudaHostAlloc((void**)&a, nBytes, cudaHostAllocPortable);
    cudaHostAlloc((void**)&b, nBytes, cudaHostAllocPortable);
    cudaHostAlloc((void**)&c, nBytes, cudaHostAllocPortable);

    printf("DEBUG2\n");

    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    printf("DEBUG3\n");

    int deviceCnt;
    cudaGetDeviceCount(&deviceCnt);

    int bytes_per_device = nBytes / deviceCnt + 1;

    float *a_d[2], *b_d[2], *c_d[2];

    dim3 grid(n/(deviceCnt*BLOCKSIZE) + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//    cudaEventRecord(start);

    printf("\nParallel starts here\n\n");

    for (int i = 0; i < deviceCnt; i++) {
        cudaSetDevice(i);
        cudaMalloc(&a_d[i], bytes_per_device);
        cudaMalloc(&b_d[i], bytes_per_device);
        cudaMalloc(&c_d[i], bytes_per_device);
    }

    for (int i = 0; i < deviceCnt; i++) {
        cudaSetDevice(i);
        cudaMemcpy(a_d[i], a, bytes_per_device, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d[i], b, bytes_per_device, cudaMemcpyHostToDevice);

        vectorAddGPU<<<block, grid>>>(a_d[i], b_d[i], c_d[i], n);

        cudaMemcpy(c, c_d[i], bytes_per_device, cudaMemcpyDeviceToHost);
    }

    for (int dev = 0; dev < deviceCnt; dev++) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    for (int i = 0; i < deviceCnt; i++) {
        cudaFree(a_d[i]);
        cudaFree(b_d[i]);
        cudaFree(c_d[i]);
    }
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
