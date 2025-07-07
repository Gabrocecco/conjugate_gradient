/* axpy_perf.c – double-precision AXPY benchmark (scalar vs. RVV)
 * ----------------------------------------------------------------------------
 * This version mirrors the "profile-isolation" pattern already used in
 * mv_random_perf.c and mv_foam_perf.c:
 *   • Compile with -DRUN_SCALAR or -DRUN_VECTOR (never both) to build only the
 *     desired kernel and its timing/CSV code – no dead branches.
 *   • All averages and CSV output fields are populated only for the kernels
 *     that were actually run; missing metrics are emitted as NaN.
 *   • Warm-up executes only the selected profile(s).
 * ----------------------------------------------------------------------------
 * Build examples:
 *   gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d -Wall -pedantic \
 *       -I../../include -DRUN_SCALAR -o ../../build/axpy_scalar \
 *       ../../src/vectorized.c ../../src/common.c axpy_perf.c
 *   gcc … -DRUN_VECTOR -o ../../build/axpy_vector …
 * ----------------------------------------------------------------------------
 * Usage:
 *   ./axpy_<profile> <num_elements>   # averages over N_TESTS runs
 */

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <inttypes.h>

#include "common.h"
#include "vectorized.h"   /* saxpy_vec_tutorial_double_vlset_opt() */

/*────────────────── compile-time profile selector ───────────────*/
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
# define N_TESTS 100        /* timed repetitions */
#endif

/*────────────────── helpers ─────────────────*/
static inline double ts_to_sec(struct timespec t)
{ return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t diff64(uint64_t s, uint64_t e)
{ return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

static inline uint64_t rdcycle64(void)
{ uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

/*────────────────── scalar kernel ─────────────────*/
static void saxpy_scalar(size_t n, double a, const double *x, double *y)
{
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

/*────────────────── main ─────────────────*/
int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_elements>\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t n = strtoull(argv[1], NULL, 10);
    if (n == 0) { fputs("<num_elements> must be > 0\n", stderr); return EXIT_FAILURE; }

    const double a = 2.0;

    /* allocate vectors */
    double *x  = malloc(n * sizeof *x);
    double *y0 = malloc(n * sizeof *y0);   /* initial y */
    double *yS = DO_SCALAR ? malloc(n * sizeof *yS) : NULL;
    double *yV = DO_VECTOR ? malloc(n * sizeof *yV) : NULL;
    assert(x && y0 && (!DO_SCALAR || yS) && (!DO_VECTOR || yV));

    srand(42);
    for (size_t i = 0; i < n; ++i) {
        x[i]  = rand() / (double)RAND_MAX;
        y0[i] = rand() / (double)RAND_MAX;
    }
    if (DO_SCALAR) memcpy(yS, y0, n * sizeof *yS);
    if (DO_VECTOR) memcpy(yV, y0, n * sizeof *yV);

    /* warm-up */
#if DO_SCALAR
    saxpy_scalar(n, a, x, yS);
    memcpy(yS, y0, n * sizeof *yS);
#endif
#if DO_VECTOR
    saxpy_vec_tutorial_double_vlset_opt(n, a, x, yV);
    memcpy(yV, y0, n * sizeof *yV);
#endif

    /* timed loops */
    struct timespec t0, t1;
    double t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0, c_v_sum = 0;

#if DO_SCALAR
    for (int it = 0; it < N_TESTS; ++it) {
        memcpy(yS, y0, n * sizeof *yS);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();

        saxpy_scalar(n, a, x, yS);

        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        t_s_sum += ts_to_sec((struct timespec){ t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec });
        c_s_sum += diff64(c0, c1);
    }
#endif
#if DO_VECTOR
    for (int it = 0; it < N_TESTS; ++it) {
        memcpy(yV, y0, n * sizeof *yV);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64();

        saxpy_vec_tutorial_double_vlset_opt(n, a, x, yV);

        uint64_t c1 = rdcycle64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        t_v_sum += ts_to_sec((struct timespec){ t1.tv_sec - t0.tv_sec, t1.tv_nsec - t0.tv_nsec });
        c_v_sum += diff64(c0, c1);
    }
#endif

    /* averages (or NaN if not run) */
    double t_s = DO_SCALAR ? t_s_sum / N_TESTS : NAN;
    double c_s = DO_SCALAR ? (double)c_s_sum / N_TESTS : NAN;
    double t_v = DO_VECTOR ? t_v_sum / N_TESTS : NAN;
    double c_v = DO_VECTOR ? (double)c_v_sum / N_TESTS : NAN;

    /* correctness */
    int pass = 1;
#if DO_SCALAR && DO_VECTOR
    for (size_t i = 0; i < n; ++i)
        if (fabs(yS[i] - yV[i]) > 1e-6) { pass = 0; break; }
#endif

    /* console */
    printf("n=%zu\n", n);
#if DO_SCALAR
    printf("scalar: time=%.6f s  cycles=%.0f\n", t_s, c_s);
#endif
#if DO_VECTOR
    printf("vector: time=%.6f s  cycles=%.0f\n", t_v, c_v);
#endif
#if DO_SCALAR && DO_VECTOR
    printf("speed-up: time=%.2f× cycles=%.2f×\n", t_s / t_v, c_s / c_v);
#endif
    printf("%s\n", pass ? "PASS" : "FAIL");

    /* CSV */
    const char *csv_path = getenv("AXPY_CSV");
    if (!csv_path) csv_path = "axpy_perf.csv";
    FILE *csv = fopen(csv_path, "a");
    if (csv) {
        if (ftell(csv) == 0) {
            fprintf(csv, "n,time_scalar,time_vector,speedup_time,cycles_scalar,cycles_vector,speedup_cycles,pass\n");
        }
        fprintf(csv, "%zu,%.6f,%.6f,%.2f,%.0f,%.0f,%.2f,%s\n",
                n, t_s, t_v, (DO_SCALAR&&DO_VECTOR)?t_s/t_v:NAN,
                c_s, c_v, (DO_SCALAR&&DO_VECTOR)?c_s/c_v:NAN,
                pass?"PASS":"FAIL");
        fclose(csv);
    }

    /* cleanup */
    free(x); free(y0);
    if (DO_SCALAR) free(yS);
    if (DO_VECTOR) free(yV);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}