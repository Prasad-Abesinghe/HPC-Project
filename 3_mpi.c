/*
 * EE7218 / EC7207 - High Performance Computing
 * Deliverable 3: Distributed Memory – MPI Matrix Multiplication
 *
 * Description:
 *   Distributes rows of matrix A across MPI processes.
 *   Each process computes its assigned block of result rows,
 *   then all blocks are gathered back to rank 0.
 *
 *   Strategy (Row-wise block distribution):
 *     - Rank 0 initialises A and B, then scatters row-blocks of A.
 *     - Every process broadcasts B (all processes need full B).
 *     - Each process computes local_C = local_A * B.
 *     - Rank 0 gathers all local_C blocks into final C.
 *
 * Compile:  mpicc -O2 -o mpi_matmul 3_mpi.c -lm
 * Run:      mpirun -np 4 ./mpi_matmul <matrix_size>
 * Example:  mpirun -np 4 ./mpi_matmul 512
 *
 * NOTE: N must be divisible by the number of processes for
 *       simplicity; add padding logic for arbitrary N if needed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ── Helpers ─────────────────────────────────────────────────── */

double *alloc_matrix(int n) {
    return (double *)calloc(n * n, sizeof(double));
}

void init_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

/* Serial reference (rank 0 only, for RMSE validation) */
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N = (argc > 1) ? atoi(argv[1]) : 256;

    if (N % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N (%d) must be divisible by nprocs (%d).\n", N, nprocs);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / nprocs;   /* rows each process owns */

    /* ── Allocate ── */
    double *A       = NULL;   /* full matrix A  – rank 0 only   */
    double *C_mpi   = NULL;   /* full result C  – rank 0 only   */
    double *C_serial= NULL;   /* serial result  – rank 0 only   */
    double *B       = alloc_matrix(N);                                    /* full B: N×N */
    double *local_A = (double *)calloc(rows_per_proc * N, sizeof(double)); /* rows_per_proc × N */
    double *local_C = (double *)calloc(rows_per_proc * N, sizeof(double));

    /* ── Rank 0 initialises data ── */
    double t_start = 0.0, t_end = 0.0;

    if (rank == 0) {
        A        = alloc_matrix(N);
        C_mpi    = alloc_matrix(N);
        C_serial = alloc_matrix(N);

        srand(42);
        init_matrix(A, N);
        init_matrix(B, N);   /* rank 0 fills B; will be broadcast */

        printf("=== MPI Matrix Multiplication ===\n");
        printf("Matrix size  : %d x %d\n", N, N);
        printf("MPI processes: %d\n", nprocs);
        printf("Rows/process : %d\n\n", rows_per_proc);

        t_start = MPI_Wtime();
    }

    /* ── Broadcast B to all processes ── */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Scatter row-blocks of A ── */
    MPI_Scatter(A,       rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ── Local computation: local_C = local_A * B ── */
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += local_A[i*N+k] * B[k*N+j];
            local_C[i*N+j] = s;
        }
    }

    /* ── Gather results to rank 0 ── */
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C_mpi,   rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        t_end = MPI_Wtime();
        double t_mpi = t_end - t_start;

        /* Serial run for comparison */
        double ts = MPI_Wtime();
        matmul_serial(A, B, C_serial, N);
        double t_serial = MPI_Wtime() - ts;

        printf("Serial time  : %.4f s\n", t_serial);
        printf("MPI time     : %.4f s\n", t_mpi);
        printf("Speedup      : %.2fx\n",  t_serial / t_mpi);
        printf("RMSE         : %.2e\n",   rmse(C_serial, C_mpi, N));

        free(A); free(C_mpi); free(C_serial);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
