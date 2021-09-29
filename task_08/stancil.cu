#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <sys/time.h>

//#define N (1024*1024*512l)
#define RADIUS 5
#define BLOCKSIZE 1024
//#define GRIDSIZE ((N-1) / BLOCKSIZE + 1)

int blockSize = BLOCKSIZE;
int gridSize;
int N;

const double EPS = 1.e-10;

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

void cudaErrorCheck() {
   // FIXME: Add code that finds the last error for CUDA functions performed.
   // Upon getting an error have it print out a meaningful error message as
   //  provided by the CUDA API, then exit with an error exit code.
}

cudaDeviceProp prop;
void getDeviceProperties() {
   // FIXME: Implement this function so as to acquire and print the following
   // device properties:
   //    Major and minor CUDA capability, total device global memory,
   //    size of shared memory per block, number of registers per block,
   //    warp size, max number of threads per block, number of multi-prccessors
   //    (SMs) per device, Maximum number of threads per block dimension (x,y,z),
   //    Maximumum number of blocks per grid dimension (x,y,z).
   //
   // These properties can be useful to dynamically optimize programs.  For
   // instance the number of SMs can be useful as a heuristic to determine
   // how many is a good number of blocks to use.  The total device global
   // memory might be important to know just how much data to operate on at
   // once.
}

void printThreadSizes() {
    printf("Vector length     = %ld (%ld MB)\n", N, N * sizeof(double) / 1024 / 1024);
    printf("Stencil radius    = %d\n\n", RADIUS);

    int noOfThreads = gridSize * blockSize;
    printf("Blocks            = %d\n", gridSize);  // no. of blocks to launch.
    printf("Threads per block = %d\n", blockSize); // no. of threads to launch.
    printf("Total threads     = %d\n", noOfThreads);
    printf("Number of grids   = %d\n", (N + noOfThreads -1)/ noOfThreads);
}

__global__ void stencil_1D (double *in, double *out, long dim) {

    long gindex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

  // Go through all data
  // Step all threads in a block to avoid synchronization problem
    while ( gindex < (dim + blockDim.x) ) {
    /* FIXME PART 2 - MODIFIY PROGRAM TO USE SHARED MEMORY. */

    // Apply the stencil
        double result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            if ( gindex + offset < dim && gindex + offset > -1)
    	        result += in[gindex + offset];
        }
    // Store the result
        if (gindex < dim)
            out[gindex] = result;

    // Update global index and quit if we are done
        gindex += stride;

        __syncthreads();
    }
}

__global__ void shared_stencil_1D(double *in, double *out, long dim) {
     __shared__ double array[BLOCKSIZE + 2 * RADIUS];

    long Idx = threadIdx.x + blockDim.x * blockIdx.x;
    int local_idx = threadIdx.x;
    array[RADIUS + local_idx] = in[Idx];

    __syncthreads();

    if ( ((local_idx + RADIUS) >= BLOCKSIZE) && ((Idx + RADIUS) < dim) ) {
        array[local_idx + 2* RADIUS] = in[Idx + RADIUS];
    }

    if ( ((local_idx - RADIUS) < 0) && ((Idx - RADIUS) > -1) ) {
        array[local_idx] = in[Idx - RADIUS];
    }

    __syncthreads();

    double result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        if (Idx + offset < dim && Idx + offset > -1)
            result += array[RADIUS + local_idx + offset];
    }
    out[Idx] = result;
}

#define True  1
#define False 0
void checkResults (double *h_in, double *h_out, int DoCheck=True) {
   // DO NOT CHANGE THIS CODE.
   // CPU calculates the stencil from data in *h_in
   // if DoCheck is True (default) it compares it with *h_out
   // to check the operation of this code.
   // If DoCheck is set to False, it can be used to time the CPU.
   int i, j, ij;
   double result;
   int err = 0;
   for (i=0; i<N; i++){  // major index.
      result = 0;
      for (j=-RADIUS; j<=RADIUS; j++){
         ij = i+j;
         if (ij>=0 && ij<N)
            result += h_in[ij];
      }
      if (DoCheck) {  // print out some errors for debugging purposes.
         if (abs(h_out[i] - result) > EPS) { // count errors.
            err++;
            if (err < 8) { // help debug
               printf("h_out[%d]=%f should be %f\n", i, h_out[i], result);
            }
         }
      } else {  // for timing purposes.
         h_out[i] = result;
      }
   }

   if (DoCheck) { // report results.
      if (err != 0) {
         printf("Error, %d elements do not match!\n", err);
      } else {
         printf("Success! All elements match CPU result.\n");
      }
   }
}

int main(int argc, char**argv) {
    assert(argc==3);
    N = atoi(argv[1]);
    bool version = atoi(argv[2]);

    gridSize = (N-1) / BLOCKSIZE + 1;

    double *h_in, *h_out;
    double *d_in, *d_out;
    long size = N * sizeof(double);
    int i;

    printThreadSizes();

    h_in = new double[N];
    h_out = new double[N];

//  getDeviceProperties();

    for (i = 0; i < N; i++)
        h_in[i] = 1;

////////////////////////////////////////////////////////
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (version == 0)
        stencil_1D<<<gridSize,blockSize>>>(d_in, d_out, N);
    else
        shared_stencil_1D<<<gridSize, blockSize, BLOCKSIZE+2*RADIUS>>>(d_in, d_out, N);
//        shared_stencil_1D<<<gridSize, blockSize, BLOCKSIZE>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "\nGPU time: " << ms << std::endl;

    checkResults(h_in, h_out);

    cudaFree(d_in);
    cudaFree(d_out);

////////////////////////////////////////////////////////
    std::cout << "Running stencil with the CPU.\n";
    double t1 = timer();
    checkResults(h_in, h_out, False);
    double t2 = timer();
    std::cout << "\nCPU Checking time: " << t2-t1 << std::endl;

    free(h_in);
    free(h_out);

    return 0;
}
