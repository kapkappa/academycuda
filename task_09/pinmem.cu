// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

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

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

//    printf("UNI: allocating memory\n");
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
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, uniStart, uniStop);
    printf("UNI: Memalloc(unified memory) + Kernel time is: %f\n", ms);
    cudaThreadSynchronize();
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

    dim3 block(256);
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

    cudaEventRecord(pinStop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, pinStart, pinStop);
    printf("PIN: Memcpy + Kernel time is: %f\n", ms);
    cudaThreadSynchronize();
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

    dim3 block(256);
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
    cudaEventCreate(&malloc_start);
    cudaEventCreate(&malloc_stop);

    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));

    printf("Copying to device..\n");

    cudaEventRecord(start);

    cudaMemcpy(a_d, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice);

    printf("Doing GPU Vector add\n");

    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("USUAL Memcpy + Kernel: %f ms\n", milliseconds);

    cudaThreadSynchronize();

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main(int argc, char **argv) {
    assert(argc==2);
    usual_sample(atoi(argv[1]));
    pinned_sample(atoi(argv[1]));
    unified_sample(atoi(argv[1]));

    return 0;
}
