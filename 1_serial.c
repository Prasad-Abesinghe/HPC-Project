/*
 * EE7218 / EC7207 - High Performance Computing
 * Deliverable 1: Serial Matrix Multiplication
 *
 * Description:
 *   Multiplies two square matrices A and B to produce C = A x B.
 *   This is the baseline (serial) version used for correctness
 *   and timing comparison.
 *
 * Compile:  gcc -O2 -o serial 1_serial.c
 * Run:      ./serial <matrix_size>
 * Example:  ./serial 512
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Allocate a flat N x N matrix */
double *alloc_matrix(int n) {
    return (double *)malloc(n * n * sizeof(double));
}

/* Fill matrix with random values in [0, 1) */
void init_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

/* C = A * B  (row-major, O(n^3)) */
void matmul_serial(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

/* Compute RMSE between two matrices (used later for parallel validation) */
double rmse(const double *ref, const double *test, int n) {
    double err = 0.0;
    for (int i = 0; i < n * n; i++) {
        double d = ref[i] - test[i];
        err += d * d;
    }
    return __builtin_sqrt(err / (n * n));
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    if (N <= 0) { fprintf(stderr, "Matrix size must be positive.\n"); return 1; }

    printf("=== Serial Matrix Multiplication ===\n");
    printf("Matrix size: %d x %d\n\n", N, N);

    srand(42);
    double *A = alloc_matrix(N);
    double *B = alloc_matrix(N);
    double *C = alloc_matrix(N);

    init_matrix(A, N);
    init_matrix(B, N);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    matmul_serial(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    printf("Elapsed time : %.4f seconds\n", elapsed);
    printf("C[0][0]      : %.6f  (spot-check)\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
