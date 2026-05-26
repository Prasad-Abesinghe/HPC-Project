# ============================================================
#  EE7218 / EC7207  –  Matrix Multiplication Project
#  Makefile for all four deliverables
# ============================================================

CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall
OMP     = -fopenmp
MATH    = -lm

all: serial openmp mpi hybrid

serial: 1_serial.c
	$(CC) $(CFLAGS) -o serial 1_serial.c $(MATH)

openmp: 2_openmp.c
	$(CC) $(CFLAGS) $(OMP) -o openmp 2_openmp.c $(MATH)

mpi: 3_mpi.c
	$(MPICC) $(CFLAGS) -o mpi_matmul 3_mpi.c $(MATH)

hybrid: 4_hybrid.c
	$(MPICC) $(CFLAGS) $(OMP) -o hybrid 4_hybrid.c $(MATH)

clean:
	rm -f serial openmp mpi_matmul hybrid

# ---- Quick test runs (N=256) ----
run_serial:
	./serial 256

run_openmp:
	./openmp 256 4

run_mpi:
	mpirun -np 4 ./mpi_matmul 256

run_hybrid:
	mpirun -np 2 ./hybrid 256 4

.PHONY: all clean run_serial run_openmp run_mpi run_hybrid
