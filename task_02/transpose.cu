#include <cuda.h>
#include <cstdlib>
#include <assert.h>

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tzp.tv_usec * 1.e-06);
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    int n = atoi(argv[1]);
    size_t size = n * n * sizeof(double);
    double *M = (double*)malloc(size);

    //init
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = i / j + j % i;
        }
    }

    double *M_t = (double*)malloc(size);

    double t1 = timer();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M_t[j][i] = M[i][j];
        }
    }

    double t2 = timer();
    std::cout << "CPU time: " << t2-t1 << std::endl;

    double *M_dev, *M_t_dev;
    cudaMalloc(&M_dev, size);
    cudaMalloc(&M_t_dev, size);

    cudaMemcpy
