/*
 * EE7218 / EC7207 - High Performance Computing
 * Deliverable 2: Shared Memory – OpenMP Matrix Multiplication
 *
 * Description:
 *   Parallelises the outer i-loop of C = A x B using OpenMP.
 *   Each thread computes a different set of rows of C.
 *
 * Compile:  gcc -O2 -fopenmp -o openmp 2_openmp.c -lm
 * Run:      ./openmp <matrix_size> <num_threads>
 * Example:  ./openmp 512 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double *alloc_matrix(int n) {
    return (double *)malloc(n * n * sizeof(double));
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

/* Serial reference (for RMSE check) */
void matmul_serial(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

/* Parallel version – rows distributed across threads */
void matmul_omp(const double *A, const double *B, double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
    }
}

double rmse(const double *ref, const double *test, int n) {
    double err = 0.0;
    for (int i = 0; i < n*n; i++) { double d = ref[i]-test[i]; err += d*d; }
    return sqrt(err / (n*n));
}

int main(int argc, char *argv[]) {
    int N       = (argc > 1) ? atoi(argv[1]) : 256;
    int threads = (argc > 2) ? atoi(argv[2]) : 4;

    omp_set_num_threads(threads);

    printf("=== OpenMP Matrix Multiplication ===\n");
    printf("Matrix size  : %d x %d\n", N, N);
    printf("Threads      : %d\n\n", threads);

    srand(42);
    double *A    = alloc_matrix(N);
    double *B    = alloc_matrix(N);
    double *C_s  = alloc_matrix(N);   /* serial result  */
    double *C_p  = alloc_matrix(N);   /* parallel result */

    init_matrix(A, N);
    init_matrix(B, N);

    /* --- Serial run --- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_serial(A, B, C_s, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_serial = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    /* --- Parallel run --- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_omp(A, B, C_p, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_omp = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    printf("Serial time  : %.4f s\n", t_serial);
    printf("OpenMP time  : %.4f s\n", t_omp);
    printf("Speedup      : %.2fx\n",  t_serial / t_omp);
    printf("RMSE         : %.2e\n",   rmse(C_s, C_p, N));

    free(A); free(B); free(C_s); free(C_p);
    return 0;
}
