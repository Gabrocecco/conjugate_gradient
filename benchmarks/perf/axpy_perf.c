// axpy_perf.c
// Build:   gcc -O3 -march=native -Wall axpy_perf.c vectorized.c -o axpy_perf
// Example: ./axpy_perf 262144   or   perf stat ./axpy_perf 262144
// If the environment variable AXPY_CSV is set, results are appended there,
// otherwise to the default "axpy_perf.csv".
// -----------------------------------------------------------------------------
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <inttypes.h>
#include <riscv_vector.h>

#include "common.h"      // must provide read_rdcycle() 64-bit counter
#include "vectorized.h"  // saxpy_vec_tutorial_double_vlset_opt()

// -----------------------------------------------------------------------------
// Reference scalar implementation: y = a * x + y
// -----------------------------------------------------------------------------
static void saxpy_scalar(size_t n, double a, const double *x, double *y)
{
    for (size_t i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

// Convert timespec to seconds (double precision)
static inline double timespec_to_sec(struct timespec t)
{
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr,
                "Usage: %s <num_elements>\n"
                "       e.g. %s 1048576\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------------------
    // Parse CLI argument
    // ---------------------------------------------------------------------
    size_t n = strtoull(argv[1], NULL, 10);
    if (n == 0) {
        fprintf(stderr, "<num_elements> must be > 0\n");
        return EXIT_FAILURE;
    }

    const double a = 2.0;

    // ---------------------------------------------------------------------
    // Allocate aligned buffers
    // ---------------------------------------------------------------------
    double *x      = aligned_alloc(64, n * sizeof(double));
    double *y_ref  = aligned_alloc(64, n * sizeof(double));
    double *y_vec  = aligned_alloc(64, n * sizeof(double));
    assert(x && y_ref && y_vec && "aligned_alloc failed");

    // Deterministic random initialization
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        x[i]     = rand() / (double)RAND_MAX;
        y_ref[i] = y_vec[i] = rand() / (double)RAND_MAX;
    }

    // ---------------------------------------------------------------------
    // 1) Scalar kernel timing & cycle count
    // ---------------------------------------------------------------------
    struct timespec ts0, ts1;
    uint64_t start_cycles, end_cycles;

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
    start_cycles = read_rdcycle();
    saxpy_scalar(n, a, x, y_ref);
    end_cycles   = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);

    const double   t_scalar = timespec_to_sec((struct timespec){ts1.tv_sec - ts0.tv_sec,
                                                                ts1.tv_nsec - ts0.tv_nsec});
    const uint64_t c_scalar = end_cycles - start_cycles;

    // ---------------------------------------------------------------------
    // 2) Vector kernel timing & cycle count
    // ---------------------------------------------------------------------
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
    start_cycles = read_rdcycle();
    saxpy_vec_tutorial_double_vlset_opt(n, a, x, y_vec);
    end_cycles   = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);

    const double   t_vector = timespec_to_sec((struct timespec){ts1.tv_sec - ts0.tv_sec,
                                                                ts1.tv_nsec - ts0.tv_nsec});
    const uint64_t c_vector = end_cycles - start_cycles;

    // ---------------------------------------------------------------------
    // 3) Correctness check (absolute tolerance 1e-6)
    // ---------------------------------------------------------------------
    int pass = 1;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(y_ref[i] - y_vec[i]) > 1e-6) {
            pass = 0;
            fprintf(stderr, "Mismatch @%zu: ref %.12f  vec %.12f\n",
                    i, y_ref[i], y_vec[i]);
            break;
        }
    }

    // ---------------------------------------------------------------------
    // 4) Print results to console
    // ---------------------------------------------------------------------
    printf("n                 : %zu\n"
           "Time  (scalar)    : %.6f s\n"
           "Time  (vectorized): %.6f s\n"
           "Speed-up (time)   : %.2fx\n\n"
           "Cycles (scalar)   : %" PRIu64 "\n"
           "Cycles (vectorized): %" PRIu64 "\n"
           "Speed-up (cycles) : %.2fx\n\n"
           "%s\n",
           n,
           t_scalar, t_vector, t_scalar / t_vector,
           c_scalar, c_vector, (double)c_scalar / (double)c_vector,
           pass ? "PASS (results match)" : "FAIL (mismatch)");

    // ---------------------------------------------------------------------
    // 5) Append results to CSV file (env-selectable path)
    // ---------------------------------------------------------------------
    const char *csv_path = getenv("AXPY_CSV");
    if (!csv_path) csv_path = "axpy_perf.csv";

    FILE *csv = fopen(csv_path, "a");
    if (!csv) {
        perror("fopen(csv_path)");
        goto cleanup;
    }

    // If file is empty, write header first
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) {
        fprintf(csv,
                "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
    }

    fprintf(csv,
            "%zu,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n,
            t_scalar, t_vector,                       // times
            t_scalar / t_vector,                      // time speed-up
            (double)c_scalar, (double)c_vector,       // cycles
            (double)c_scalar / (double)c_vector,      // cycle speed-up
            pass ? "PASS" : "FAIL");

    fclose(csv);

cleanup:
    // ---------------------------------------------------------------------
    // 6) Free buffers & exit
    // ---------------------------------------------------------------------
    free(x);
    free(y_ref);
    free(y_vec);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}