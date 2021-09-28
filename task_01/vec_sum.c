#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>

double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
};

int main (int argc, char**argv) {
    if (argc != 2) { return 1; }
    int n = atoi(argv[1]);
    //host vecs
    double *h_a, *h_b, *h_c;

    size_t bytes = n * sizeof(double);

    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    int i;
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

double t1 = timer();

    for (i = 0; i < n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

double t2 = timer();

    printf("computation time is: %f\n", t2-t1);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
