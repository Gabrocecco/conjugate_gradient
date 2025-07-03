/* axpy_perf.c  – double-precision version
   Build:
     gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
         -Wall -pedantic -I../../include \
         -o /conjugate_gradient_laptop/build/axpy_perf \
         ../../src/vectorized.c ../../src/common.c axpy_perf.c
  
   Usage:
     ./axpy_perf <num_elements>
     (The program runs the scalar and vector kernels N_TESTS times and prints / appends the average.)
  
   Configuration: N_TESTS controls the number of timed repetitions.
*/

// -----------------------------------------------------------------------------
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <inttypes.h>
#include <riscv_vector.h>
#include "common.h"
#include "vectorized.h"  // saxpy_vec_tutorial_double_vlset_opt()

#ifndef N_TESTS
#define N_TESTS 100        // number of timed repetitions
#endif

// -----------------------------------------------------------------------------
// aligned_alloc fallback for toolchains without C11 aligned_alloc()
// -----------------------------------------------------------------------------
// #if (__STDC_VERSION__ < 201112L) || (defined(__APPLE__) && !defined(aligned_alloc))
// static void *aligned_malloc(size_t alignment, size_t size)
// {
//     void *ptr = NULL;
//     if (posix_memalign(&ptr, alignment, size) != 0) ptr = NULL;
//     return ptr;
// }
// #define aligned_alloc(alignment, size) aligned_malloc(alignment, size)
// #endif

static inline double timespec_to_sec(struct timespec t){return t.tv_sec + t.tv_nsec * 1e-9;}

static inline uint64_t read_rdcycle(void)
{
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline uint64_t diff64(uint64_t s, uint64_t e) { return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

// -----------------------------------------------------------------------------
// Scalar double-precision AXPY: y = a * x + y
// -----------------------------------------------------------------------------
static void saxpy_scalar(size_t n, double a, const double *x, double *y)
{
    for (size_t i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_elements>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t n = strtoull(argv[1], NULL, 10);
    if (n == 0) {
        fprintf(stderr, "<num_elements> must be > 0\n");
        return EXIT_FAILURE;
    }

    const double a = 2.0;

    // ---------------------------------------------------------------------
    // Allocate buffers
    // ---------------------------------------------------------------------
    // double *x       = aligned_alloc(64, n * sizeof(double));
    // double *y_init  = aligned_alloc(64, n * sizeof(double));
    // double *y_ref   = aligned_alloc(64, n * sizeof(double));
    // double *y_vec   = aligned_alloc(64, n * sizeof(double));

    double *x = malloc(n * sizeof(double));
    double *y_init = malloc(n * sizeof(double));
    double *y_ref = malloc(n * sizeof(double));
    double *y_vec = malloc(n * sizeof(double));

    assert(x && y_init && y_ref && y_vec);

    srand(42);
    for (size_t i = 0; i < n; ++i) {
        x[i]      = rand() / (double)RAND_MAX;
        y_init[i] = rand() / (double)RAND_MAX;
    }

    memcpy(y_ref, y_init, n * sizeof(double));
    memcpy(y_vec, y_init, n * sizeof(double));

    // Warm-up
    saxpy_scalar(n, a, x, y_ref);
    memcpy(y_ref, y_init, n * sizeof(double));
    saxpy_vec_tutorial_double_vlset_opt(n, a, x, y_vec);  // RVV kernel (double)
    memcpy(y_vec, y_init, n * sizeof(double));

    // ---------------------------------------------------------------------
    // Timed repetitions
    // ---------------------------------------------------------------------
    double   t_scalar_sum = 0.0, t_vector_sum = 0.0;
    uint64_t c_scalar_sum = 0,    c_vector_sum = 0;

    struct timespec ts0, ts1;

    for (int iter = 0; iter < N_TESTS; ++iter) {
        // Scalar
        memcpy(y_ref, y_init, n * sizeof(double));
        uint64_t start_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
        saxpy_scalar(n, a, x, y_ref);
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        uint64_t end_cycles = read_rdcycle();

        t_scalar_sum += timespec_to_sec((struct timespec){ts1.tv_sec - ts0.tv_sec,
                                                         ts1.tv_nsec - ts0.tv_nsec});
        // c_scalar_sum += end_cycles - start_cycles;
        c_scalar_sum += diff64(start_cycles, end_cycles);

        // Vector
        memcpy(y_vec, y_init, n * sizeof(double));
        start_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
        saxpy_vec_tutorial_double_vlset_opt(n, a, x, y_vec);
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        end_cycles = read_rdcycle();

        t_vector_sum += timespec_to_sec((struct timespec){ts1.tv_sec - ts0.tv_sec,
                                                         ts1.tv_nsec - ts0.tv_nsec});
        // c_vector_sum += end_cycles - start_cycles;
        c_vector_sum += diff64(start_cycles, end_cycles);
    }

    const double   t_scalar_avg = t_scalar_sum / N_TESTS;
    const double   t_vector_avg = t_vector_sum / N_TESTS;
    const uint64_t c_scalar_avg = c_scalar_sum / N_TESTS;
    const uint64_t c_vector_avg = c_vector_sum / N_TESTS;

    // Correctness check
    int pass = 1;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(y_ref[i] - y_vec[i]) > 1e-6) { pass = 0; break; }
    }

    // Console output
    printf("n                 : %zu (average of %d runs)\n"
           "Time  (scalar)    : %.6f s\n"
           "Time  (vectorized): %.6f s\n"
           "Speed-up (time)   : %.2fx\n\n"
           "Cycles (scalar)   : %" PRIu64 "\n"
           "Cycles (vectorized): %" PRIu64 "\n"
           "Speed-up (cycles) : %.2fx\n\n"
           "%s\n",
           n, N_TESTS,
           t_scalar_avg, t_vector_avg, t_scalar_avg / t_vector_avg,
           c_scalar_avg, c_vector_avg, (double)c_scalar_avg / (double)c_vector_avg,
           pass ? "PASS (results match)" : "FAIL (mismatch)");

    // CSV handling
    const char *csv_path = getenv("AXPY_CSV");
    if (!csv_path) csv_path = "axpy_perf.csv";

    FILE *csv = fopen(csv_path, "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0) {
            fprintf(csv,
                    "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
        }
        fprintf(csv,
                "%zu,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
                n,
                t_scalar_avg, t_vector_avg,
                t_scalar_avg / t_vector_avg,
                (double)c_scalar_avg, (double)c_vector_avg,
                (double)c_scalar_avg / (double)c_vector_avg,
                pass ? "PASS" : "FAIL");
        fclose(csv);
    }

    free(x);
    free(y_init);
    free(y_ref);
    free(y_vec);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
