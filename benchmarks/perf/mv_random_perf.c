/*  ───────────────────────────────────────────────────────────────────────────
    random_mv_perf.c   –  RV64  (no flush)
    Bench: symmetric sparse ELL x vector, scalar vs. RVV-VLSET
    Build:
      gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
          -Wall -pedantic -I../../include \
          -o ../../build/random_mv_perf \
          ../../src/vectorized.c ../../src/common.c ../../src/random.c \
          ../../src/ell.c random_mv_perf.c
    ─────────────────────────────────────────────────────────────────────────── 
*/

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
#include "vectorized.h" /* mv_ell_symmetric_full_colmajor_vector_vlset_opt() */
#include "coo.h"
#include "csr.h"

/* -------------------------- Parameters --------------------------------*/
#define N_TESTS 100

/* -------------------------- Utility inline --------------------------------*/
static inline double ts_to_sec(struct timespec t) { return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t diff64(uint64_t s, uint64_t e) { return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

/* Read 64 bit (RV64) cycle counter */
static inline uint64_t rdcycle64(void)
{
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

/* -------------------------- Core benchmark -------------------------- */
static void mv_rvv_vs_scalar(int n, double sparsity)
{
    /* CSV ------------------------------------------------------------------ */
    const char *csv_path = getenv("MV_RND_CSV"); // the name of the CSV file can be set via environment variable in the run_random_mv_perf.sh script
    if (!csv_path)
        csv_path = "mv_prof_random.csv"; // default name
    FILE *out = fopen(csv_path, "a");
    assert(out && "Unable to open CSV output file");

    if (ftell(out) == 0) // if the file is empty, write the header. This benchamrk is run multiple times with different parameters, so we want to keep the header only once.
        fprintf(out,
                "n,sparsity,max_nnz_row,time_serial,time_vectorized,speedup_time,"
                "cycles_serial,cycles_vector,speedup_cycles,pass\n");

    /* Random COO ----------------------------------------------------------- */
    int upper_nnz;
    double *coo_vals, *diag;
    int *coo_i, *coo_j;
    generate_sparse_symmetric_coo(n, sparsity,                                   // generate a random sparse summetric matrix in COO format with given sparisty
                                  &upper_nnz, &coo_vals, &coo_i, &coo_j, &diag); // if sparsity id 0.01 each offdiagonal element has a 1% chance of being non-zero

    int max_nnz = compute_max_nnz_row_full(n, upper_nnz, coo_i, coo_j); // compute the maximum number of non-zero elements in each row of the matrix (not counting diagonal elements)

    /* COO → ELL ------------------------------------------------------------ */
    double *ell_val = calloc((size_t)n * max_nnz, sizeof(*ell_val)); // allocate memory for ELL arrays
    int *ell_col = malloc((size_t)n * max_nnz * sizeof(*ell_col));
    coo_to_ell_symmetric_full_colmajor( // convert the COO format to ELL format
        n, upper_nnz, coo_vals, coo_i, coo_j, ell_val, ell_col, max_nnz);

    // we convert the column indices to 64-bit integers to use them with the RVV vectorized function
    uint64_t *ell_col64 = malloc((size_t)n * max_nnz * sizeof(*ell_col64));
    for (size_t k = 0; k < (size_t)n * max_nnz; ++k)
        ell_col64[k] = (uint64_t)ell_col[k];

    /* Vectors -------------------------------------------------------------- */
    double *x = malloc(n * sizeof(*x));
    double *y_s = calloc(n, sizeof(*y_s));
    double *y_v = calloc(n, sizeof(*y_v));
    for (int i = 0; i < n; ++i)
        x[i] = rand() / (double)RAND_MAX;

    /* Accumulators --------------------------------------------------------- */
    struct timespec t0, t1;
    double t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0, c_v_sum = 0;

    // Warm-up run 
    mv_ell_symmetric_full_colmajor_sdtint(n, max_nnz, diag, ell_val, ell_col64, x, y_s);
    memset(y_v, 0, n * sizeof(*y_v));
    mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, max_nnz, diag, ell_val, ell_col64, x, y_v);

    /* ---------- Scalar loop ---------- */
    for (int it = 0; it < N_TESTS; ++it)
    {
        memset(y_s, 0, n * sizeof(*y_s));

        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();
        mv_ell_symmetric_full_colmajor_sdtint(
            n, max_nnz, diag, ell_val, ell_col64, x, y_s);
        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_s_sum += ts_to_sec((struct timespec){// accumulate the time in seconds used by the scalar function
                                               t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});

        // c_s_sum += c1 - c0; // accumulate the number of cycles used by the scalar function
        c_s_sum += diff64(c0, c1); // this is a workaround to avoid the compiler optimizing away the rdcycle64() calls
    }

    /* ---------- Vector loop ---------- */
    for (int it = 0; it < N_TESTS; ++it)
    {
        memset(y_v, 0, n * sizeof(*y_v));

        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();
        mv_ell_symmetric_full_colmajor_vector_vlset_opt(
            n, max_nnz, diag, ell_val, ell_col64, x, y_v);
        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_v_sum += ts_to_sec((struct timespec){
            t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});
        // c_v_sum += c1 - c0;
        c_v_sum += diff64(c0, c1);
    }

    double t_s = t_s_sum / N_TESTS;
    double t_v = t_v_sum / N_TESTS;
    double c_s = (double)c_s_sum / N_TESTS;
    double c_v = (double)c_v_sum / N_TESTS;

    /* Correctness ---------------------------------------------------------- */
    int pass = 1;
    for (int i = 0; i < n; ++i)
        if (!fp_eq(y_s[i], y_v[i], 1e-6))
        {
            pass = 0;
            break;
        }

    /* printf the results to the console ---------------------------- */
    printf("n = %d, sparsity = %.4f, max_nnz_row = %d, "
           "time_serial = %.6f, time_vectorized = %.6f, speedup_time = %.2f, "
           "cycles_serial = %.2f, cycles_vector = %.2f, speedup_cycles = %.2f, %s\n",
           n, sparsity, max_nnz,
           t_s, t_v, t_s / t_v,
           c_s, c_v, c_s / c_v,
           pass ? "PASS" : "FAIL");

    /* CSV output ----------------------------------------------------------- */
    fprintf(out,
            "%d,%.4f,%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n, sparsity, max_nnz,
            t_s, t_v, t_s / t_v,
            c_s, c_v, c_s / c_v,
            pass ? "PASS" : "FAIL");
    fclose(out);

    /* Cleanup -------------------------------------------------------------- */
    free(coo_vals);
    free(coo_i);
    free(coo_j);
    free(diag);
    free(ell_val);
    free(ell_col);
    free(ell_col64);
    free(x);
    free(y_s);
    free(y_v);
}

/* ─────────────────────────────── main ─────────────────────────────── */
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <n> <sparsity 0-1>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);

    srand(42);
    mv_rvv_vs_scalar(n, sparsity);
    return 0;
}
