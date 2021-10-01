// High level matrix multiplication on GPU using CUDA with Thrust, CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <sys/time.h>
double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const thrust::device_vector<float> &A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char**argv) {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
    assert(argc==2);
    int n = atoi(argv[1]);
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = n;

	thrust::device_vector<float> d_A(nr_rows_A * nr_cols_A), d_B(nr_rows_B * nr_cols_B), d_C(nr_rows_C * nr_cols_C);

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(thrust::raw_pointer_cast(&d_A[0]), nr_rows_A, nr_cols_A);
	GPU_fill_rand(thrust::raw_pointer_cast(&d_B[0]), nr_rows_B, nr_cols_B);

	// Optionally we can print the data
//	std::cout << "A =" << std::endl;
//	print_matrix(d_A, nr_rows_A, nr_cols_A);
//	std::cout << "B =" << std::endl;
//	print_matrix(d_B, nr_rows_B, nr_cols_B);

	// Multiply A and B on GPU
    double t1 = timer();
	gpu_blas_mmul(thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_B[0]), thrust::raw_pointer_cast(&d_C[0]), nr_rows_A, nr_cols_A, nr_cols_B);
    double t2 = timer();
    std::cout << "Sgemm time: " << t2-t1 << std::endl;

    double it = n / 1024;
    double oper_num = 2 * it * it * it;
    std::cout << "Efficiency: " << oper_num / (t2-t1) << " GFLOPS" << std::endl;

	//Print the result
//	std::cout << "C =" << std::endl;
//	print_matrix(d_C, nr_rows_C, nr_cols_C);

	return 0;
}
