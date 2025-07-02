/*  ───────────────────────────────────────────────────────────────────────────
    random_mv_perf.c
    Bench: symmetric sparse ELL × vector, scalar vs. RVV-optimized (VLSET)

    Build (esempio):
      gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
          -Wall -pedantic -I../../include \
          -o ../../build/random_mv_perf \
          ../../src/vectorized.c ../../src/common.c ../../src/random.c \
          ../../src/ell.c random_mv_perf.c
    ─────────────────────────────────────────────────────────────────────────── */

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
#include "vectorized.h"   /* mv_ell_symmetric_full_colmajor_vector_vlset_opt() */
#include "coo.h"       /* generate_sparse_symmetric_coo() */
#include "csr.h"
/* ───────────────────────── Globali / parametri ──────────────────────────── */
#define CACHE_FLUSH_ELEMS  (100 * 1000 * 1000)      /* 100 M int ≈ 400 MB */
static int garbage[CACHE_FLUSH_ELEMS];

#define N_TESTS 100                                 /* ripetizioni per la media */

/* ──────────────────────────── Utility inline ────────────────────────────── */
static inline double ts_to_sec(struct timespec t)
{ return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t rdcycle(void)
{ uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

static inline uint64_t diff64(uint64_t s, uint64_t e)
{ return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

/* ------------------------------------------------------------------------- */
/*  Cache-flush: percorre 400 MB per invalidare L1/L2/LLC.                   */
/* ------------------------------------------------------------------------- */
static inline void flush_cache_by_accessing_garbage(void)
{
    /* ≥ 64 MB di footprint per sicurezza */
    static_assert(CACHE_FLUSH_ELEMS * sizeof(garbage[0]) >= 64 * 1000 * 1000,
                  "garbage[] troppo piccolo per il flush della cache");

    for (size_t i = 0; i < CACHE_FLUSH_ELEMS; ++i)
        garbage[i] += (int)i;          /* write-allocate & read-modify-write */
}

/* ────────────────────────── Core benchmark ─────────────────────────────── */
static void mv_rvv_vs_scalar(int n, double sparsity, int n_tests)
{
    /* -- CSV -- */
    const char *csv_path = getenv("MV_RND_CSV");
    if (!csv_path) csv_path = "mv_prof_random.csv";
    FILE *out = fopen(csv_path, "a");
    assert(out && "Unable to open CSV output file");

    if (ftell(out) == 0)
        fprintf(out,
            "n,sparsity,max_nnz_row,time_serial,time_vectorized,speedup_time,"
            "cycles_serial,cycles_vector,speedup_cycles,pass\n");

    /* -- random COO matrix -- */
    int upper_nnz;
    double *coo_vals, *diag;
    int *coo_i, *coo_j;
    generate_sparse_symmetric_coo(n, sparsity,
                                  &upper_nnz, &coo_vals, &coo_i, &coo_j, &diag);

    int max_nnz = compute_max_nnz_row_full(n, upper_nnz, coo_i, coo_j);

    /* -- convert → ELL (col-major, symmetric full storage) -- */
    double *ell_val = calloc((size_t)n * max_nnz, sizeof(*ell_val));
    int    *ell_col = malloc((size_t)n * max_nnz * sizeof(*ell_col));

    coo_to_ell_symmetric_full_colmajor(
        n, upper_nnz, coo_vals, coo_i, coo_j, ell_val, ell_col, max_nnz);

    uint64_t *ell_col64 = malloc((size_t)n * max_nnz * sizeof(*ell_col64));
    for (size_t k = 0; k < (size_t)n * max_nnz; ++k)
        ell_col64[k] = (uint64_t)ell_col[k];

    /* -- vectors -- */
    double *x   = malloc(n * sizeof(*x));
    double *y_s = calloc(n, sizeof(*y_s));
    double *y_v = calloc(n, sizeof(*y_v));
    for (int i = 0; i < n; ++i) x[i] = rand() / (double)RAND_MAX;

    /* -- timing accumulators -- */
    struct timespec t0, t1;
    uint64_t c0, c1;
    double   t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0,  c_v_sum = 0;

    /* ---------- Scalar loop ---------- */
    for (int it = 0; it < n_tests; ++it) {
        flush_cache_by_accessing_garbage();
        memset(y_s, 0, n * sizeof(*y_s));

        clock_gettime(CLOCK_MONOTONIC_RAW, &t0); c0 = rdcycle();
        mv_ell_symmetric_full_colmajor_sdtint(
            n, max_nnz, diag, ell_val, ell_col64, x, y_s);
        c1 = rdcycle(); clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_s_sum += ts_to_sec((struct timespec){
                        t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});
        c_s_sum += diff64(c0, c1);
    }

    /* ---------- Vector loop ---------- */
    for (int it = 0; it < n_tests; ++it) {
        flush_cache_by_accessing_garbage();
        memset(y_v, 0, n * sizeof(*y_v));

        clock_gettime(CLOCK_MONOTONIC_RAW, &t0); c0 = rdcycle();
        mv_ell_symmetric_full_colmajor_vector_vlset_opt(
            n, max_nnz, diag, ell_val, ell_col64, x, y_v);
        c1 = rdcycle(); clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_v_sum += ts_to_sec((struct timespec){
                        t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec});
        c_v_sum += diff64(c0, c1);
    }

    double t_s = t_s_sum / n_tests;
    double t_v = t_v_sum / n_tests;
    double c_s = (double)c_s_sum / n_tests;
    double c_v = (double)c_v_sum / n_tests;

    /* ---------- correctness ---------- */
    int pass = 1;
    for (int i = 0; i < n; ++i)
        if (!fp_eq(y_s[i], y_v[i], 1e-6)) { pass = 0; break; }

    /* ---------- CSV output ---------- */
    fprintf(out,
            "%d,%.4f,%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n, sparsity, max_nnz,
            t_s, t_v, t_s / t_v,
            c_s, c_v, c_s / c_v,
            pass ? "PASS" : "FAIL");
    fclose(out);

    /* ---------- cleanup ---------- */
    free(coo_vals); free(coo_i); free(coo_j); free(diag);
    free(ell_val);  free(ell_col);  free(ell_col64);
    free(x); free(y_s); free(y_v);
}

/* ─────────────────────────────── main ──────────────────────────────── */
int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <n> <sparsity 0-1>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int    n        = atoi(argv[1]);
    double sparsity = atof(argv[2]);

    srand(42);
    mv_rvv_vs_scalar(n, sparsity, N_TESTS);
    return 0;
}
