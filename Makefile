# ============================================================
#  EE7218 / EC7207  –  Matrix Multiplication Project
#  Makefile for all four deliverables
# ============================================================

CC      = gcc
CFLAGS  = -O2 -Wall
OMP     = -fopenmp
MATH    = -lm

# MS-MPI paths (Windows / MinGW)
MPI_INC = "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
MPI_LIB = "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64"

EXE_EXT = .exe

all: serial openmp mpi hybrid

serial: 1_serial.c
	$(CC) $(CFLAGS) -o serial$(EXE_EXT) 1_serial.c $(MATH)

openmp: 2_openmp.c
	$(CC) $(CFLAGS) $(OMP) -o openmp$(EXE_EXT) 2_openmp.c $(MATH)

mpi: 3_mpi.c libmsmpi.a
	$(CC) $(CFLAGS) -I$(MPI_INC) -L. -o mpi_matmul$(EXE_EXT) 3_mpi.c -lmsmpi $(MATH)

hybrid: 4_hybrid.c libmsmpi.a
	$(CC) $(CFLAGS) $(OMP) -I$(MPI_INC) -L. -o hybrid$(EXE_EXT) 4_hybrid.c -lmsmpi $(MATH)

# Generate MinGW-compatible import library from msmpi.dll
libmsmpi.a:
	gendef "C:/Windows/System32/msmpi.dll"
	dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
	del msmpi.def 2>nul || true

clean:
	del /Q serial.exe openmp.exe mpi_matmul.exe hybrid.exe libmsmpi.a msmpi.def 2>nul || true

# ---- Quick test runs (N=256) ----
run_serial:
	serial.exe 256

run_openmp:
	openmp.exe 256 4

run_mpi:
	mpiexec -n 4 mpi_matmul.exe 256

run_hybrid:
	mpiexec -n 2 hybrid.exe 256 4

.PHONY: all clean run_serial run_openmp run_mpi run_hybrid
