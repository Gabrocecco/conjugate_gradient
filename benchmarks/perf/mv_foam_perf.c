/* mv_foam_perf.c  – symmetric ELL x vector
   Build:
     gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
         -Wall -pedantic -I../../include \
         -o /conjugate_gradient_laptop/build/mv_foam_perf \
         ../../src/vectorized.c ../../src/common.c mv_foam_perf.c
  
   Usage:
     ./mv_foam_perf <openfoam_system_file>   # averaged over N_TESTS runs
*/

// Configuration
#ifndef N_TESTS
#define N_TESTS 100
#endif

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <assert.h>

#include "ell.h"
#include "common.h"
#include "vectorized.h" // mv_ell_symmetric_full_colmajor_vector_vlset_opt()
#include "parser.h"   // parseDoubleArray / parseIntArray 

static inline double ts_to_sec(struct timespec t) { return t.tv_sec + 1e-9 * t.tv_nsec; }

/* Read 64 bit (RV64) cycle counter */
static inline uint64_t rdcycle64(void)
{
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
static inline uint64_t diff64(uint64_t s, uint64_t e) { return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <openfoam.system>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *fname = argv[1];
    FILE *file = fopen(fname, "r");
    if (!file)
    {
        perror("fopen");
        return EXIT_FAILURE;
    }

    // read OpenFOAM matrix data
    int n = 0, n_upper = 0, n_lower = 0;
    double *diag = parseDoubleArray(file, "diag", &n);
    double *upper = parseDoubleArray(file, "upper", &n_upper);  
    int *coo_cols = parseIntArray(file, "upperAddr", &n_upper);
    int *coo_rows = parseIntArray(file, "lowerAddr", &n_lower);
    fclose(file);

    // compute the max number of non-zero elements in each row (not counting diagonal elements)
    int nnz_max = compute_max_nnz_row_full(n, n_upper, coo_rows, coo_cols);

    // convert COO to ELL format
    double *ell_val = aligned_alloc(64, (size_t)nnz_max * n * sizeof(double));
    int *ell_col = aligned_alloc(64, (size_t)nnz_max * n * sizeof(int));
    coo_to_ell_symmetric_full_colmajor(n, n_upper, upper, coo_rows, coo_cols, ell_val, ell_col, nnz_max);

    // convert ELL column indices to 64-bit integers for vectorized operations
    uint64_t *ell_col64 = aligned_alloc(64, (size_t)nnz_max * n * sizeof(uint64_t));
    for (size_t k = 0; k < (size_t)nnz_max * n; k++)
        ell_col64[k] = (uint64_t)ell_col[k];

    // allocate input and output vectors
    double *x = aligned_alloc(64, n * sizeof(double));
    double *y_ref = calloc(n, sizeof(double));
    double *y_vec = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
        x[i] = rand() / (double)RAND_MAX;

    // warm‑up
    mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_val, ell_col64, x, y_ref);
    memset(y_vec, 0, n * sizeof(double));
    mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, nnz_max, diag, ell_val, ell_col64, x, y_vec);

    // timing
    double t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0, c_v_sum = 0;
    struct timespec t0, t1;
    for (int it = 0; it < N_TESTS; ++it)
    {
        memset(y_ref, 0, n * sizeof(double));
        uint64_t cs = rdcycle();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_val, ell_col64, x, y_ref);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        uint64_t ce = rdcycle();
        t_s_sum += ts_to_sec((struct timespec){t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});
        c_s_sum += diff64(cs, ce);
        // c_s_sum += ce - cs;

        memset(y_vec, 0, n * sizeof(double));
        cs = rdcycle();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, nnz_max, diag, ell_val, ell_col64, x, y_vec);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        ce = rdcycle();
        t_v_sum += ts_to_sec((struct timespec){t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});
        c_v_sum += diff64(cs, ce);
    }

    double t_s = t_s_sum / N_TESTS, t_v = t_v_sum / N_TESTS;
    uint64_t c_s = c_s_sum / N_TESTS, c_v = c_v_sum / N_TESTS;
    
    int pass = 1;
    for (int i = 0; i < n; i++)
        if (fabs(y_ref[i] - y_vec[i]) > 1e-6)
        {
            pass = 0;
            break;
        }

    printf("n = %d, nnz_max = %d, time_s = %.6f, time_v = %.6f, speedup_t = %.2f, cycles_s = %" PRIu64 ", cycles_v = %" PRIu64 ", speedup_c = %.2f, %s\n",
           n, nnz_max, t_s, t_v, t_s / t_v, c_s, c_v, (double)c_s / (double)c_v, pass ? "PASS" : "FAIL");

    const char *csv = getenv("MV_CSV");
    if (!csv)
        csv = "mv_foam_perf.csv";
    FILE *out = fopen(csv, "a");
    if (out)
    {
        fseek(out, 0, SEEK_END);
        if (ftell(out) == 0)
            fprintf(out,
                    "n,nnz_max,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
        fprintf(out, "%d,%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n", n, nnz_max, t_s, t_v, t_s / t_v, (double)c_s, (double)c_v, (double)c_s / (double)c_v, pass ? "PASS" : "FAIL");
        fclose(out);
    }

    free(x);
    free(y_ref);
    free(y_vec);
    free(diag);
    free(upper);
    free(coo_cols);
    free(coo_rows);
    free(ell_val);
    free(ell_col);
    free(ell_col64);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
