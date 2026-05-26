# HPC Matrix Multiplication Project
### EE7218 / EC7207 — High Performance Computing

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Technologies & Tools](#4-technologies--tools)
5. [Project Structure](#5-project-structure)
6. [Implementation Details](#6-implementation-details)
   - [Deliverable 1 — Serial](#61-deliverable-1--serial-baseline)
   - [Deliverable 2 — OpenMP](#62-deliverable-2--openmp-shared-memory)
   - [Deliverable 3 — MPI](#63-deliverable-3--mpi-distributed-memory)
   - [Deliverable 4 — Hybrid](#64-deliverable-4--hybrid-mpi--openmp)
7. [Environment Setup](#7-environment-setup)
8. [Build Instructions](#8-build-instructions)
9. [Run Instructions](#9-run-instructions)
10. [Performance Results](#10-performance-results)
11. [Correctness Validation](#11-correctness-validation)

---

## 1. Project Overview

This project implements square matrix multiplication (C = A × B) using four progressively
parallel approaches. The goal is to measure the speedup gained at each level of parallelism
and to understand the trade-offs between shared-memory and distributed-memory programming models.

| Deliverable | File | Parallelism Model |
|---|---|---|
| 1 | `1_serial.c` | None (baseline) |
| 2 | `2_openmp.c` | Shared memory (OpenMP) |
| 3 | `3_mpi.c` | Distributed memory (MPI) |
| 4 | `4_hybrid.c` | Hybrid (MPI + OpenMP) |

---

## 2. Problem Statement

Given two N×N matrices A and B filled with random `double` values, compute:

```
C[i][j] = Σ(k=0 to N-1) A[i][k] * B[k][j]
```

The naive algorithm has **O(N³)** time complexity. Parallelism does not change the
algorithmic complexity but reduces wall-clock time by distributing work across cores/processes.

All matrices are stored as **flat row-major 1D arrays** in heap memory:
```
Element [i][j]  →  array[i * N + j]
```

---

## 3. Architecture

### 3.1 Serial Architecture

```
┌─────────────────────────────┐
│         Single Process      │
│                             │
│   A[N×N]  B[N×N]  C[N×N]  │
│                             │
│   Triple nested loop        │
│   i → j → k                │
└─────────────────────────────┘
```

### 3.2 OpenMP Architecture (Shared Memory)

```
┌──────────────────────────────────────────────┐
│             Single Process (OS)              │
│                                              │
│  Shared Memory: A[N×N], B[N×N], C[N×N]      │
│                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│  │Thread│ │Thread│ │Thread│ │Thread│  ...    │
│  │  0   │ │  1   │ │  2   │ │  3   │        │
│  │rows  │ │rows  │ │rows  │ │rows  │        │
│  │0..k  │ │k..2k │ │2k..3k│ │3k..N │        │
│  └──────┘ └──────┘ └──────┘ └──────┘        │
│                                              │
│  #pragma omp parallel for schedule(static)  │
└──────────────────────────────────────────────┘
```

- All threads share the same A, B, C in memory.
- The outer `i` loop is divided equally among threads (`schedule(static)`).
- No synchronisation needed — each thread writes to a distinct row of C.

### 3.3 MPI Architecture (Distributed Memory)

```
                  ┌─────────────────────┐
                  │       Rank 0        │
                  │  Init A[N×N], B[N×N]│
                  └────────┬────────────┘
                           │
            ┌──────────────┼──────────────┐
            │  MPI_Bcast(B)│              │
            ▼              ▼              ▼
       ┌─────────┐   ┌─────────┐   ┌─────────┐
       │ Rank 0  │   │ Rank 1  │   │ Rank P-1│
       │local_A  │   │local_A  │   │local_A  │
       │(N/P rows│   │(N/P rows│   │(N/P rows│
       │of A)    │   │of A)    │   │of A)    │
       │full B   │   │full B   │   │full B   │
       │local_C  │   │local_C  │   │local_C  │
       └────┬────┘   └────┬────┘   └────┬────┘
            │             │             │
            └──────────── ▼ ────────────┘
                   MPI_Gather(local_C)
                          │
                   ┌──────▼──────┐
                   │   Rank 0    │
                   │  C[N×N]     │
                   │  (assembled)│
                   └─────────────┘
```

**MPI Communication pattern:**
- `MPI_Bcast` — broadcast full matrix B to all ranks (every rank needs all columns of B).
- `MPI_Scatter` — distribute row-blocks of A (each rank gets N/P rows).
- `MPI_Gather` — collect computed row-blocks of C back to rank 0.

### 3.4 Hybrid Architecture (MPI + OpenMP)

```
┌────────────────────────────────────────────────────┐
│  Node / MPI Rank 0                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │OMP T 0 │ │OMP T 1 │ │OMP T 2 │ │OMP T 3 │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│          Owns rows  0  ..  N/2-1                    │
└────────────────────────────────────────────────────┘
         MPI_Scatter ↑↓ MPI_Gather
┌────────────────────────────────────────────────────┐
│  Node / MPI Rank 1                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │OMP T 0 │ │OMP T 1 │ │OMP T 2 │ │OMP T 3 │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│          Owns rows  N/2 .. N-1                      │
└────────────────────────────────────────────────────┘

Total parallelism = MPI processes × OpenMP threads per process
```

- MPI handles coarse-grained distribution across processes/nodes.
- OpenMP handles fine-grained parallelism within each process using `#pragma omp parallel for`.
- `MPI_Init_thread` is used with `MPI_THREAD_FUNNELED` — only the master thread makes MPI calls.

---

## 4. Technologies & Tools

| Component | Technology | Purpose |
|---|---|---|
| Language | C (C99) | All implementations |
| Serial compiler | GCC (MinGW-W64 14.2) | Compile serial and OpenMP |
| OpenMP | GCC built-in (`-fopenmp`) | Shared-memory threading |
| MPI runtime | Microsoft MPI (MS-MPI) | Distributed-memory processes |
| MPI compiler wrapper | GCC + MS-MPI SDK | Replaces `mpicc` on Windows |
| MPI launcher | `mpiexec.exe` | Launch multi-process MPI jobs |
| Build system | GNU Make (via MinGW) | Automate compilation |
| Math library | libm (`-lm`) | `sqrt()` for RMSE |
| Timing (serial) | `clock_gettime(CLOCK_MONOTONIC)` | High-resolution wall time |
| Timing (MPI) | `MPI_Wtime()` | Portable MPI timer |

### Key Compiler Flags

| Flag | Effect |
|---|---|
| `-O2` | Level-2 optimisation (loop unrolling, inlining) |
| `-Wall` | Enable all warnings |
| `-fopenmp` | Enable OpenMP pragmas and runtime |
| `-lm` | Link math library |
| `-lmsmpi` | Link MS-MPI shared library |
| `-I<path>` | Add MS-MPI include directory |
| `-L.` | Search current directory for `libmsmpi.a` |

---

## 5. Project Structure

```
New one/
├── 1_serial.c              # Deliverable 1 — serial baseline
├── 2_openmp.c              # Deliverable 2 — OpenMP shared memory
├── 3_mpi.c                 # Deliverable 3 — MPI distributed memory
├── 4_hybrid.c              # Deliverable 4 — MPI + OpenMP hybrid
├── Makefile                # Build and run rules
├── PROJECT_DOCUMENTATION.md  # This document
├── Analysis_Report.docx    # Written analysis report
└── Project Guideline.pdf   # Assignment specification
```

**Generated at build time (not committed to git):**
```
├── serial.exe
├── openmp.exe
├── mpi_matmul.exe
├── hybrid.exe
└── libmsmpi.a              # MinGW import library for MS-MPI
```

---

## 6. Implementation Details

### 6.1 Deliverable 1 — Serial (Baseline)

**File:** `1_serial.c`

**Algorithm:** Classic triple-nested loop, O(N³).

```c
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += A[i*N+k] * B[k*N+j];
        C[i*N+j] = sum;
    }
```

**Key points:**
- Fixed random seed (`srand(42)`) for reproducibility.
- Timing uses `clock_gettime(CLOCK_MONOTONIC)` — not affected by system clock adjustments.
- Output: elapsed time and a spot-check value `C[0][0]`.

**Usage:**
```
serial.exe <N>
```

---

### 6.2 Deliverable 2 — OpenMP (Shared Memory)

**File:** `2_openmp.c`

**Algorithm:** Parallelise the outer `i` loop with a single OpenMP pragma.

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < N; k++)
            s += A[i*N+k] * B[k*N+j];
        C[i*N+j] = s;
    }
}
```

**Key points:**
- `schedule(static)` divides rows into equal contiguous chunks — ideal when all rows take equal time.
- Runs both serial and parallel back-to-back in the same process and computes RMSE between results.
- Thread count passed as a command-line argument (`omp_set_num_threads`).
- No race conditions: each thread writes to distinct rows of C.

**Usage:**
```
openmp.exe <N> <num_threads>
```

---

### 6.3 Deliverable 3 — MPI (Distributed Memory)

**File:** `3_mpi.c`

**Algorithm:** Row-wise block distribution across MPI processes.

**Step-by-step execution:**

| Step | MPI Call | Description |
|---|---|---|
| 1 | `MPI_Init` | Initialise MPI environment |
| 2 | `MPI_Comm_rank/size` | Each process learns its rank and total count |
| 3 | Rank 0 only | Allocate + initialise full A and B |
| 4 | `MPI_Bcast(B)` | Broadcast all of B to every rank |
| 5 | `MPI_Scatter(A)` | Send N/P rows of A to each rank |
| 6 | Local compute | Each rank computes its rows of C |
| 7 | `MPI_Gather(C)` | Rank 0 assembles complete C |
| 8 | Rank 0 only | Validate with serial run, print speedup |
| 9 | `MPI_Finalize` | Clean up MPI |

**Constraint:** N must be divisible by the number of processes.

**Memory layout per rank:**

| Buffer | Who allocates | Size |
|---|---|---|
| `A` | Rank 0 only | N × N doubles |
| `B` | All ranks | N × N doubles |
| `local_A` | All ranks | (N/P) × N doubles |
| `local_C` | All ranks | (N/P) × N doubles |
| `C_mpi` | Rank 0 only | N × N doubles |

**Usage:**
```
mpiexec -n <num_processes> mpi_matmul.exe <N>
```

---

### 6.4 Deliverable 4 — Hybrid (MPI + OpenMP)

**File:** `4_hybrid.c`

**Algorithm:** MPI distributes row-blocks; OpenMP parallelises computation within each block.

```c
// Each MPI rank runs this block:
#pragma omp parallel for schedule(static)
for (int i = 0; i < rows_per_proc; i++) {
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < N; k++)
            s += local_A[i*N+k] * B[k*N+j];
        local_C[i*N+j] = s;
    }
}
```

**Key differences from pure MPI:**
- Uses `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)` instead of `MPI_Init` — required when mixing MPI and OpenMP threads.
- Thread count passed as a second command-line argument.
- Total parallelism = `num_processes × threads_per_process`.

**Usage:**
```
mpiexec -n <num_processes> hybrid.exe <N> <threads_per_process>
```

---

## 7. Environment Setup

### 7.1 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| GCC (MinGW-W64) | 14.2+ | Includes `gendef`, `dlltool`, OpenMP support |
| MS-MPI Runtime | 10.x | `mpiexec.exe` — from Microsoft |
| MS-MPI SDK | 10.x | Headers + import libraries |
| GNU Make | any | Bundled with MinGW |

### 7.2 Install GCC (MinGW-W64)

Download from: https://winlibs.com or via MSYS2.

Ensure `gcc`, `make`, `gendef`, and `dlltool` are on your `PATH`.

### 7.3 Install MS-MPI

1. Download **MS-MPI runtime** (`msmpisetup.exe`) — installs `mpiexec.exe`
2. Download **MS-MPI SDK** (`msmpisdk.msi`) — installs headers and `.lib` files

Both are available from the Microsoft GitHub releases:
https://github.com/microsoft/Microsoft-MPI/releases

Default install paths:
- Runtime: `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe`
- SDK headers: `C:\Program Files (x86)\Microsoft SDKs\MPI\Include\`
- SDK libraries: `C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\`

### 7.4 Generate MinGW Import Library for MS-MPI

MS-MPI ships a MSVC `.lib` file, not a MinGW `.a` file. You must generate one:

```powershell
cd "D:\Academic\SEM 07\HPC\New one"
gendef "C:/Windows/System32/msmpi.dll"
dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
del msmpi.def
```

This `libmsmpi.a` must be present in the project directory before compiling MPI or hybrid targets.
The Makefile's `mpi` and `hybrid` targets handle this automatically.

---

## 8. Build Instructions

### Build all targets

```powershell
cd "D:\Academic\SEM 07\HPC\New one"
make all
```

This produces: `serial.exe`, `openmp.exe`, `mpi_matmul.exe`, `hybrid.exe`

### Build individual targets

```powershell
make serial        # only 1_serial.c
make openmp        # only 2_openmp.c
make mpi           # only 3_mpi.c  (also generates libmsmpi.a if needed)
make hybrid        # only 4_hybrid.c
```

### Manual compile commands (without Make)

```powershell
# Serial
gcc -O2 -Wall -o serial.exe 1_serial.c -lm

# OpenMP
gcc -O2 -Wall -fopenmp -o openmp.exe 2_openmp.c -lm

# Generate MinGW MPI import library (one-time)
gendef "C:/Windows/System32/msmpi.dll"
dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll

# MPI
gcc -O2 -Wall -I"C:/Program Files (x86)/Microsoft SDKs/MPI/Include" -L. -o mpi_matmul.exe 3_mpi.c -lmsmpi -lm

# Hybrid
gcc -O2 -Wall -fopenmp -I"C:/Program Files (x86)/Microsoft SDKs/MPI/Include" -L. -o hybrid.exe 4_hybrid.c -lmsmpi -lm
```

### Clean build artefacts

```powershell
make clean
```

---

## 9. Run Instructions

### Deliverable 1 — Serial

```powershell
./serial.exe <N>

# Examples:
./serial.exe 256     # 256×256 matrix
./serial.exe 512     # 512×512 matrix
./serial.exe 1024    # 1024×1024 matrix
```

**Output:**
```
=== Serial Matrix Multiplication ===
Matrix size: 256 x 256

Elapsed time : 0.0147 seconds
C[0][0]      : 63.606115  (spot-check)
```

---

### Deliverable 2 — OpenMP

```powershell
./openmp.exe <N> <num_threads>

# Examples:
./openmp.exe 256 2     # 2 threads
./openmp.exe 256 4     # 4 threads
./openmp.exe 512 8     # 8 threads
```

**Output:**
```
=== OpenMP Matrix Multiplication ===
Matrix size  : 256 x 256
Threads      : 4

Serial time  : 0.0131 s
OpenMP time  : 0.0046 s
Speedup      : 2.86x
RMSE         : 0.00e+00
```

**Note:** `num_threads` should not exceed the number of logical CPU cores.

---

### Deliverable 3 — MPI

```powershell
mpiexec -n <num_processes> ./mpi_matmul.exe <N>

# Examples:
mpiexec -n 2 ./mpi_matmul.exe 256
mpiexec -n 4 ./mpi_matmul.exe 256
mpiexec -n 4 ./mpi_matmul.exe 512
```

**Constraint:** N must be divisible by num_processes (e.g., N=256 with 4 processes: 256/4=64 rows each).

**Output:**
```
=== MPI Matrix Multiplication ===
Matrix size  : 256 x 256
MPI processes: 4
Rows/process : 64

Serial time  : 0.0131 s
MPI time     : 0.0065 s
Speedup      : 2.01x
RMSE         : 0.00e+00
```

---

### Deliverable 4 — Hybrid

```powershell
mpiexec -n <num_processes> ./hybrid.exe <N> <threads_per_process>

# Examples:
mpiexec -n 2 ./hybrid.exe 256 4     # 2 procs × 4 threads = 8 total
mpiexec -n 4 ./hybrid.exe 512 2     # 4 procs × 2 threads = 8 total
```

**Constraint:** N must be divisible by num_processes.

**Output:**
```
=== Hybrid (MPI + OpenMP) Matrix Multiplication ===
Matrix size      : 256 x 256
MPI processes    : 2
Threads/process  : 4
Total threads    : 8

Serial time  : 0.0133 s
Hybrid time  : 0.0044 s
Speedup      : 3.02x
RMSE         : 0.00e+00
```

---

### Quick test via Makefile targets

```powershell
make run_serial     # runs serial.exe 256
make run_openmp     # runs openmp.exe 256 4
make run_mpi        # runs mpiexec -n 4 mpi_matmul.exe 256
make run_hybrid     # runs mpiexec -n 2 hybrid.exe 256 4
```

---

## 10. Performance Results

Results measured on Windows 11, AMD/Intel CPU, matrix size N=256.

| Implementation | Config | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| Serial | 1 core | 0.0131 | 1.00x | 100% |
| OpenMP | 4 threads | 0.0046 | 2.86x | 71.4% |
| MPI | 4 processes | 0.0065 | 2.01x | 50.3% |
| Hybrid | 2 proc × 4 threads | 0.0044 | 3.02x | 75.5% |

**Efficiency** = Speedup / num_parallel_units × 100%

### Observations

- **OpenMP** achieves better efficiency than MPI at this matrix size because shared-memory has
  negligible communication overhead compared to MPI's broadcast + scatter + gather.

- **MPI** overhead is significant for small N (N=256) because the time spent on
  `MPI_Bcast` + `MPI_Scatter` + `MPI_Gather` is comparable to the compute time.
  MPI becomes more efficient for larger N (N≥1024) where communication cost is amortised.

- **Hybrid** achieves the best overall speedup by combining both levels of parallelism.
  It benefits from fewer MPI processes (less communication overhead) while using threads
  to exploit all available cores within each process.

---

## 11. Correctness Validation

All parallel implementations are validated against the serial result using **RMSE**
(Root Mean Square Error):

```
RMSE = sqrt( Σ(C_serial[i] - C_parallel[i])² / N² )
```

An RMSE of `0.00e+00` (or below floating-point epsilon ~1e-15) confirms that the
parallel result is numerically identical to the serial result.

This validation runs automatically every time you run the OpenMP, MPI, or hybrid executables.

---

*Document generated for EE7218 / EC7207 High Performance Computing — 2026*
