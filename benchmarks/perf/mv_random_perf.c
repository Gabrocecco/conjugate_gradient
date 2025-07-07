/* ──────────────────────────────────────────────────────────────
   random_mv_perf.c        • RV64  (no flush, no “full” mode)
   Benchmark: symmetric sparse ELL × vector
               – scalar   vs.  RVV-VLSET
   Build examples (see run script):
     gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
         -Wall -pedantic -I../../include \
         -DRUN_SCALAR -o ../../build/mv_scalar random_mv_perf.c ...
     gcc … -DRUN_VECTOR -o ../../build/mv_vector  random_mv_perf.c ...
   ----------------------------------------------------------------
   Compile-time profile selector
   •  -DRUN_SCALAR    → only scalar loop is built
   •  -DRUN_VECTOR    → only vector loop is built
   Never define both and never run a “full” build.
   ---------------------------------------------------------------- */

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>      /* NAN */
#include <time.h>
#include <inttypes.h>
#include <assert.h>

#include "ell.h"
#include "common.h"
#include "vectorized.h"   /* mv_ell_symmetric_full_colmajor_vector_vlset_opt() */
#include "coo.h"
#include "csr.h"

/* ────────────────── profile flags ────────────────── */
#if defined(RUN_SCALAR) && defined(RUN_VECTOR)
# error "Define either RUN_SCALAR or RUN_VECTOR, not both"
#endif
#if !defined(RUN_SCALAR) && !defined(RUN_VECTOR)
# error "Define at least one profile macro"
#endif

/* ─────────── profile flags (numeric) ─────────── */
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

/* ────────────────── parameters ────────────────────── */
#define N_TESTS 100

/* ────────────────── helpers ───────────────────────── */
static inline double ts_to_sec(struct timespec t)
{ return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t diff64(uint64_t s, uint64_t e)
{ return e >= s ? e - s : UINT64_MAX - s + 1 + e; }

/* rdcycle / rdinstret – 64-bit on RV64 */
static inline uint64_t rdcycle64(void)
{ uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

static inline uint64_t rdinstret64(void)
{ uint64_t v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v; }

/* ────────────────── benchmark core ────────────────── */
static void mv_rvv_vs_scalar(int n, double sparsity)
{
    /* CSV ---------------------------------------------------------- */
    const char *csv_path = getenv("MV_RND_CSV");
    if (!csv_path) csv_path = "mv_prof_random.csv";
    FILE *out = fopen(csv_path, "a");
    assert(out && "Unable to open CSV output file");

    if (ftell(out) == 0)      /* write header once */
        fprintf(out,
          "n,sparsity,max_nnz_row,"
          "time_serial,time_vectorized,speedup_time,"
          "cycles_serial,cycles_vector,speedup_cycles,"
          "inst_serial,inst_vector,speedup_inst,"
          "ipc_serial,ipc_vector,"
          "pass\n");

    /* ---- build random symmetric matrix in COO -------------------- */
    int upper_nnz;
    double *coo_vals, *diag;
    int *coo_i, *coo_j;
    generate_sparse_symmetric_coo(n, sparsity,
                                  &upper_nnz, &coo_vals, &coo_i, &coo_j, &diag);

    int max_nnz = compute_max_nnz_row_full(n, upper_nnz, coo_i, coo_j);

    /* ---- convert to ELL (column-major) --------------------------- */
    double *ell_val = calloc((size_t)n * max_nnz, sizeof(*ell_val));
    int    *ell_col = malloc((size_t)n * max_nnz * sizeof(*ell_col));
    coo_to_ell_symmetric_full_colmajor(
        n, upper_nnz, coo_vals, coo_i, coo_j, ell_val, ell_col, max_nnz);

    /* convert indices to 64 bit for RVV kernel */
    uint64_t *ell_col64 = malloc((size_t)n * max_nnz * sizeof(*ell_col64));
    for (size_t k = 0; k < (size_t)n * max_nnz; ++k)
        ell_col64[k] = (uint64_t)ell_col[k];

    /* ---- vectors ------------------------------------------------- */
    double *x  = malloc(n * sizeof(*x));
    double *yS = calloc(n, sizeof(*yS));
    double *yV = calloc(n, sizeof(*yV));
    for (int i = 0; i < n; ++i) x[i] = rand() / (double)RAND_MAX;

    /* ---- accumulators ------------------------------------------- */
    struct timespec t0, t1;
    double   t_s_sum = 0., t_v_sum = 0.;
    uint64_t c_s_sum = 0, c_v_sum = 0;
    uint64_t i_s_sum = 0, i_v_sum = 0;

/* --------------------------------------------------------------
   warm up   (run only the kernels that will be benchmarked)
   -------------------------------------------------------------- */
#if DO_SCALAR
    mv_ell_symmetric_full_colmajor_sdtint(
        n, max_nnz, diag, ell_val, ell_col64, x, yS);
#endif

#if DO_VECTOR
    memset(yV, 0, n * sizeof(*yV));   /* keep symmetry with the test loop */
    mv_ell_symmetric_full_colmajor_vector_vlset_opt(
        n, max_nnz, diag, ell_val, ell_col64, x, yV);
#endif

    /* ---------- scalar loop ---------- */
#if DO_SCALAR
    for (int it = 0; it < N_TESTS; ++it) {
        memset(yS, 0, n*sizeof(*yS));
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64(), i0 = rdinstret64();

        mv_ell_symmetric_full_colmajor_sdtint(n,max_nnz,diag,ell_val,ell_col64,x,yS);

        uint64_t c1 = rdcycle64(), i1 = rdinstret64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_s_sum += ts_to_sec((struct timespec){t1.tv_sec-t0.tv_sec, t1.tv_nsec-t0.tv_nsec});
        c_s_sum += diff64(c0, c1);
        i_s_sum += diff64(i0, i1);
    }
#endif

    /* ---------- vector loop ---------- */
#if DO_VECTOR
    for (int it = 0; it < N_TESTS; ++it) {
        memset(yV, 0, n*sizeof(*yV));
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        uint64_t c0 = rdcycle64(), i0 = rdinstret64();

        mv_ell_symmetric_full_colmajor_vector_vlset_opt(n,max_nnz,diag,ell_val,ell_col64,x,yV);

        uint64_t c1 = rdcycle64(), i1 = rdinstret64();
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        t_v_sum += ts_to_sec((struct timespec){t1.tv_sec-t0.tv_sec, t1.tv_nsec-t0.tv_nsec});
        c_v_sum += diff64(c0, c1);
        i_v_sum += diff64(i0, i1);
    }
#endif

    /* ---- averages (or NAN if not run) --------------------------- */
    double t_s = DO_SCALAR ? t_s_sum / N_TESTS : NAN;
    double c_s = DO_SCALAR ? (double)c_s_sum / N_TESTS : NAN;
    double inst_s = DO_SCALAR ? (double)i_s_sum / N_TESTS : NAN;

    double t_v = DO_VECTOR ? t_v_sum / N_TESTS : NAN;
    double c_v = DO_VECTOR ? (double)c_v_sum / N_TESTS : NAN;
    double inst_v = DO_VECTOR ? (double)i_v_sum / N_TESTS : NAN;

    double ipc_s = inst_s / c_s;          /* will propagate NAN if c_s is NAN */
    double ipc_v = inst_v / c_v;

    /* ---- correctness check (only if both results exist) --------- */
    int pass = 1;
#if DO_SCALAR && DO_VECTOR
    for (int i = 0; i < n; ++i)
        if (!fp_eq(yS[i], yV[i], 1e-6)) { pass = 0; break; }
#endif

    /* ---- console output ----------------------------------------- */
    printf("n=%d\n"
        "spars=%.4f\n"
        "max=%d\n"
        "tS=%.6f\n"
        "tV=%.6f\n"
        "cS=%.0f\n"
        "cV=%.0f\n"
        "iS=%.0f\n"
        "iV=%.0f\n"
        "ipcS=%.3f\n"
        "ipcV=%.3f\n"
        "%s\n",
        n, sparsity, max_nnz,
        t_s, t_v, c_s, c_v, inst_s, inst_v, ipc_s, ipc_v,
        pass ? "PASS" : "FAIL");

    /* ---- CSV output --------------------------------------------- */
    fprintf(out,
        "%d,%.4f,%d,"
        "%.6f,%.6f,NaN,"     /* speed-ups left for post-processing */
        "%.0f,%.0f,NaN,"
        "%.0f,%.0f,NaN,"
        "%.3f,%.3f,"
        "%s\n",
        n, sparsity, max_nnz,
        t_s, t_v,
        c_s, c_v,
        inst_s, inst_v,
        ipc_s, ipc_v,
        pass ? "PASS" : "FAIL");
    fclose(out);

    /* ---- cleanup ------------------------------------------------ */
    free(coo_vals); free(coo_i); free(coo_j); free(diag);
    free(ell_val);  free(ell_col); free(ell_col64);
    free(x); free(yS); free(yV);
}

/* ──────────────────────── main ─────────────────────── */
int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr,"Usage: %s <n> <sparsity 0-1>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);

    srand(42);
    mv_rvv_vs_scalar(n, sparsity);
    return 0;
}
