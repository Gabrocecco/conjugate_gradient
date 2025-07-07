/* mv_foam_perf.c – symmetric ELL × vector benchmark (OpenFOAM matrices)
 * -----------------------------------------------------------------------------
 * This version aligns mv_foam_perf.c with the profiling structure already
 * adopted in mv_random_perf.c:
 *   • Compile–time selector macros RUN_SCALAR / RUN_VECTOR let you build either
 *     the scalar or the vectorized kernel only (never both in the same binary),
 *     avoiding any branch mis‑predictions or dead code in tight loops.
 *   • DO_SCALAR / DO_VECTOR evaluate to 1 or 0 so the whole warm‑up, timing and
 *     validation code is conditionally compiled away – zero overhead when a
 *     profile is disabled.
 *   • All statistics (time and cycles) are computed separately and a short CSV
 *     line is appended to $MV_CSV (default: mv_foam_perf.csv).
 * -----------------------------------------------------------------------------
 * Build examples (see also run script):
 *   gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d -Wall -pedantic \
 *       -I../../include -DRUN_SCALAR   -o ../../build/mv_scalar  mv_foam_perf.c \
 *       ../../src/vectorized.c ../../src/common.c
 *   gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d -Wall -pedantic \
 *       -I../../include -DRUN_VECTOR   -o ../../build/mv_vector  mv_foam_perf.c \
 *       ../../src/vectorized.c ../../src/common.c
 * ---------------------------------------------------------------------------*/

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
#include "vectorized.h"  /* mv_ell_symmetric_full_colmajor_vector_vlset_opt() */
#include "parser.h"      /* parseDoubleArray / parseIntArray */

/*────────────────── compile‑time profile selector ─────────────────*/
#if defined(RUN_SCALAR) && defined(RUN_VECTOR)
# error "Define either RUN_SCALAR or RUN_VECTOR, not both"
#endif
#if !defined(RUN_SCALAR) && !defined(RUN_VECTOR)
# error "Define at least one profile macro (RUN_SCALAR or RUN_VECTOR)"
#endif

#ifdef RUN_SCALAR
# define DO_SCALAR 1
#else
# define DO_SCALAR 0
#endif
#ifdef RUN_VECTOR
# define DO_VECTOR 1
#else
# define DO_VECTOR 0
#endif

#ifndef N_TESTS
# define N_TESTS 100
#endif

/*────────────────── helpers ─────────────────*/
static inline double ts_to_sec(struct timespec t)
{ return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t diff64(uint64_t s, uint64_t e)
{ return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

/* Read 64‑bit (RV64) cycle counter */
static inline uint64_t rdcycle64(void)
{ uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

/*────────────────── main ─────────────────*/
int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <openfoam.system>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *fname = argv[1];
    FILE *file = fopen(fname, "r");
    if (!file) { perror("fopen"); return EXIT_FAILURE; }

    /*──────────────── read OpenFOAM matrix ────────────────*/
    int n = 0, n_upper = 0, n_lower = 0;
    double *diag   = parseDoubleArray(file, "diag",      &n);
    double *upper  = parseDoubleArray(file, "upper",     &n_upper);
    int    *coo_c  = parseIntArray   (file, "upperAddr", &n_upper);
    int    *coo_r  = parseIntArray   (file, "lowerAddr", &n_lower);
    fclose(file);

    int nnz_max = compute_max_nnz_row_full(n, n_upper, coo_r, coo_c);

    /* convert to ELL (column‑major) */
    double   *ell_val  = aligned_alloc(64, (size_t)nnz_max * n * sizeof *ell_val);
    int      *ell_col  = aligned_alloc(64, (size_t)nnz_max * n * sizeof *ell_col);
    coo_to_ell_symmetric_full_colmajor(n, n_upper, upper, coo_r, coo_c,
                                       ell_val, ell_col, nnz_max);
    uint64_t *ell_col64 = aligned_alloc(64, (size_t)nnz_max * n * sizeof *ell_col64);
    for (size_t k = 0, tot = (size_t)nnz_max * n; k < tot; ++k) ell_col64[k] = (uint64_t)ell_col[k];

    /* vectors */
    double *x  = aligned_alloc(64, n * sizeof *x);
    double *yS = DO_SCALAR ? calloc(n, sizeof *yS) : NULL;
    double *yV = DO_VECTOR ? calloc(n, sizeof *yV) : NULL;
    srand(42);
    for (int i = 0; i < n; ++i) x[i] = rand() / (double)RAND_MAX;

    /*──────────────── warm‑up (only run kernels that matter) */
#if DO_SCALAR
    mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_val, ell_col64, x, yS);
#endif
#if DO_VECTOR
    mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, nnz_max, diag, ell_val, ell_col64, x, yV);
#endif

    /*──────────────── timed loops ─────────────────*/
    struct timespec t0, t1;
    double t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0, c_v_sum = 0;

#if DO_SCALAR
    for (int it = 0; it < N_TESTS; ++it) {
        memset(yS, 0, n * sizeof *yS);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();

        mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_val, ell_col64, x, yS);

        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        t_s_sum += ts_to_sec((struct timespec){ t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec });
        c_s_sum += diff64(c0, c1);
    }
#endif
#if DO_VECTOR
    for (int it = 0; it < N_TESTS; ++it) {
        memset(yV, 0, n * sizeof *yV);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();

        mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, nnz_max, diag, ell_val, ell_col64, x, yV);

        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        t_v_sum += ts_to_sec((struct timespec){ t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec });
        c_v_sum += diff64(c0, c1);
    }
#endif

    /* averages */
    double t_s = DO_SCALAR ? t_s_sum / N_TESTS : NAN;
    double c_s = DO_SCALAR ? (double)c_s_sum / N_TESTS : NAN;
    double t_v = DO_VECTOR ? t_v_sum / N_TESTS : NAN;
    double c_v = DO_VECTOR ? (double)c_v_sum / N_TESTS : NAN;

    /* correctness check */
    int pass = 1;
#if DO_SCALAR && DO_VECTOR
    for (int i = 0; i < n; ++i)
        if (fabs(yS[i] - yV[i]) > 1e-6) { pass = 0; break; }
#endif

    printf("n=%d, nnz_max=%d\n", n, nnz_max);
#if DO_SCALAR
    printf("scalar:   time=%.6f s  cycles=%.0f\n", t_s, c_s);
#endif
#if DO_VECTOR
    printf("vector:   time=%.6f s  cycles=%.0f\n", t_v, c_v);
#endif
#if DO_SCALAR && DO_VECTOR
    printf("speed‑up: time=%.2f× cycles=%.2f×\n", t_s / t_v, c_s / c_v);
#endif
    printf("%s\n", pass ? "PASS" : "FAIL");

    /*──────────────── CSV output ────────────────*/
    const char *csv = getenv("MV_CSV");
    if (!csv) csv = "mv_foam_perf.csv";
    FILE *out = fopen(csv, "a");
    if (out) {
        if (ftell(out) == 0) {
            fprintf(out, "n,nnz_max,time_scalar,time_vector,speedup_time,cycles_scalar,cycles_vector,speedup_cycles,pass\n");
        }
        fprintf(out, "%d,%d,%.6f,%.6f,%.2f,%.0f,%.0f,%.2f,%s\n",
                n, nnz_max,
                t_s, t_v, (DO_SCALAR && DO_VECTOR) ? t_s / t_v : NAN,
                c_s, c_v, (DO_SCALAR && DO_VECTOR) ? c_s / c_v : NAN,
                pass ? "PASS" : "FAIL");
        fclose(out);
    }

    /* cleanup */
    free(x);
    if (DO_SCALAR) free(yS);
    if (DO_VECTOR) free(yV);
    free(diag); free(upper); free(coo_c); free(coo_r);
    free(ell_val); free(ell_col); free(ell_col64);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
