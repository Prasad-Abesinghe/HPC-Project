/*
 * EE7218 / EC7207 - High Performance Computing
 * Deliverable 4: Hybrid Programming – MPI + OpenMP Matrix Multiplication
 *
 * Description:
 *   Combines distributed memory (MPI) and shared memory (OpenMP).
 *   MPI distributes row-blocks of A across nodes; within each
 *   process, OpenMP threads parallelise the computation of those rows.
 *
 *   Parallelism levels:
 *     Level 1 (MPI)    – inter-process: each MPI rank owns N/nprocs rows
 *     Level 2 (OpenMP) – intra-process: threads divide those rows further
 *
 * Compile:  mpicc -O2 -fopenmp -o hybrid 4_hybrid.c -lm
 * Run:      mpirun -np 2 ./hybrid <matrix_size> <threads_per_proc>
 * Example:  mpirun -np 2 ./hybrid 512 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

/* ── Helpers ─────────────────────────────────────────────────── */

double *alloc_matrix(int n) {
    return (double *)calloc(n, sizeof(double));
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

void matmul_serial(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

double rmse(const double *ref, const double *test, int n) {
    double err = 0.0;
    for (int i = 0; i < n*n; i++) { double d=ref[i]-test[i]; err+=d*d; }
    return sqrt(err / (double)(n*n));
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    /* MPI_THREAD_FUNNELED: only the master thread calls MPI */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED.\n");
        MPI_Finalize(); return 1;
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N       = (argc > 1) ? atoi(argv[1]) : 256;
    int threads = (argc > 2) ? atoi(argv[2]) : 2;

    if (N % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by nprocs (%d).\n", N, nprocs);
        MPI_Finalize(); return 1;
    }

    omp_set_num_threads(threads);

    int rows_per_proc = N / nprocs;

    double *A        = NULL;
    double *C_hybrid = NULL;
    double *C_serial = NULL;
    double *B        = alloc_matrix(N * N);
    double *local_A  = alloc_matrix(rows_per_proc * N);
    double *local_C  = alloc_matrix(rows_per_proc * N);

    double t_start = 0.0;

    if (rank == 0) {
        A        = alloc_matrix(N * N);
        C_hybrid = alloc_matrix(N * N);
        C_serial = alloc_matrix(N * N);

        srand(42);
        init_matrix(A, N * N);
        init_matrix(B, N * N);

        printf("=== Hybrid (MPI + OpenMP) Matrix Multiplication ===\n");
        printf("Matrix size      : %d x %d\n", N, N);
        printf("MPI processes    : %d\n", nprocs);
        printf("Threads/process  : %d\n", threads);
        printf("Total threads    : %d\n\n", nprocs * threads);

        t_start = MPI_Wtime();
    }

    /* Broadcast B */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter rows of A */
    MPI_Scatter(A,       rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ── Local computation with OpenMP ── */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += local_A[i*N+k] * B[k*N+j];
            local_C[i*N+j] = s;
        }
    }

    /* Gather results */
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C_hybrid, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double t_hybrid = MPI_Wtime() - t_start;

        double ts = MPI_Wtime();
        matmul_serial(A, B, C_serial, N);
        double t_serial = MPI_Wtime() - ts;

        printf("Serial time  : %.4f s\n", t_serial);
        printf("Hybrid time  : %.4f s\n", t_hybrid);
        printf("Speedup      : %.2fx\n",  t_serial / t_hybrid);
        printf("RMSE         : %.2e\n",   rmse(C_serial, C_hybrid, N));

        free(A); free(C_hybrid); free(C_serial);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
