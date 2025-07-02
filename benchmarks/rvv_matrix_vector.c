#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <assert.h>
#include <time.h>
#include <inttypes.h>
#include "common.h"
#include "vectorized.h"
#include "ell.h"
#include "coo.h"
#include "csr.h"
#include "parser.h"
#include "mmio.h"

static inline uint64_t read_rdcycle()
{
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

const size_t n = 100 * 1000 * 1000;

#define N 100 * 1000 * 1000
int garbage[N];

void flush_cache_by_accessing_garbage()
{
    assert(n * sizeof(garbage[0]) >= 64 * 1000 * 1000);
    for (size_t i = 0; i < n; ++i)
    {
        garbage[i] += i;
    }
}

void mv_rvv_vs_scalar(int n, double sparsity, int N_TESTS)
{
    // --- Open file and write CSV header if file is empty ---
    FILE *out = fopen("scripts/data/mv_prof_random_O3_avg_tail_opt.csv", "a");
    assert(out && "Unable to open output file");

    // Check if file is empty, then write header
    fseek(out, 0, SEEK_END);
    if (ftell(out) == 0)
    {
        fprintf(out, "n,sparsity,max_nnz_row,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
    }

    printf("Running with n = %d, sparsity = %.4f\n", n, sparsity);

    // --- Generate random COO ---
    int upper_nnz;
    double *coo_vals, *diag;
    int *coo_i, *coo_j;
    generate_sparse_symmetric_coo(n, sparsity, &upper_nnz, &coo_vals, &coo_i, &coo_j, &diag);

    int max_nnz = compute_max_nnz_row_full(n, upper_nnz, coo_i, coo_j);
    printf("n = %d, upper_nnz = %d, max_nnz_row = %d\n", n, upper_nnz, max_nnz);

    // --- Convert to ELL ---
    double *ell_values = calloc((size_t)n * max_nnz, sizeof(*ell_values));
    int *ell_cols = malloc((size_t)n * max_nnz * sizeof(*ell_cols));
    coo_to_ell_symmetric_full_colmajor(n, upper_nnz, coo_vals, coo_i, coo_j, ell_values, ell_cols, max_nnz);

    // --- Analyze ELL matrix ---
    analyze_ell_matrix_full_colmajor(n, max_nnz, ell_values, ell_cols);

    // --- Allocate vectors ---
    double *x = malloc(n * sizeof(*x));
    double *y = calloc(n, sizeof(*y));
    double *y_vectorized = calloc(n, sizeof(*y_vectorized));
    for (int i = 0; i < n; ++i)
        x[i] = rand() / (double)RAND_MAX;

    // --- Convert column indices to uint64_t ---
    uint64_t *ell_cols64 = malloc((size_t)n * max_nnz * sizeof(*ell_cols64));
    for (int i = 0; i < n * max_nnz; ++i)
        ell_cols64[i] = (uint64_t)ell_cols[i];

    // repeat test for N_TESTS times
    struct timespec start, end;
    uint64_t start_cycles, end_cycles;
    uint64_t cycles_serial = 0;
    uint64_t cycles_vector = 0;
    double time_serial = 0.0;
    double time_vector = 0.0;

    // --- Serial ---
    for (int i = 0; i < N_TESTS; i++)
    {
        //! flush cache
        flush_cache_by_accessing_garbage();
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        mv_ell_symmetric_full_colmajor_sdtint(n, max_nnz, diag, ell_values, ell_cols64, x, y);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_serial += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_serial += end_cycles - start_cycles;
    }

    // take average of N_TESTS
    const double cycles_serial_mean = (double)cycles_serial / N_TESTS;
    const double time_serial_mean = (double)time_serial / N_TESTS;

    // --- Vectorized ---
    for (int i = 0; i < N_TESTS; i++)
    {
        //! flush cache
        flush_cache_by_accessing_garbage();
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        // mv_ell_symmetric_full_colmajor_vector(n, max_nnz, diag, ell_values, ell_cols64, x, y_vectorized);
        mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, max_nnz, diag, ell_values, ell_cols64, x, y_vectorized);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_vector += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_vector += end_cycles - start_cycles;
    }

    // take average on N_TESTS
    const double cycles_vector_mean = (double)cycles_vector / N_TESTS;
    const double time_vector_mean = (double)time_vector / N_TESTS;

    // --- Compare results ---
    int pass = 1;
    for (int i = 0; i < n; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6))
        {
            pass = 0;
            break;
        }
    }

    // --- Report ---
    printf("Time serial     : %.6f s\n", time_serial_mean);
    printf("Time vectorized : %.6f s\n", time_vector_mean);
    printf("Speedup (time)  : %.2fx\n", time_serial_mean / time_vector_mean);
    printf("Cycles serial   : %.2f\n", cycles_serial_mean);
    printf("Cycles vector   : %.2f \n", cycles_vector_mean);
    printf("Speedup (cycles): %.2fx\n", cycles_serial_mean / cycles_vector_mean);
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // --- Save to CSV ---
    fprintf(out, "%d,%.4f,%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n, sparsity, max_nnz,
            time_serial_mean, time_vector_mean,
            time_serial_mean / time_vector_mean,
            cycles_serial_mean, cycles_vector_mean,
            (double)cycles_serial_mean / (double)cycles_vector_mean,
            pass ? "PASS" : "FAIL");

    fclose(out);

    // --- Cleanup ---
    free(coo_vals);
    free(coo_i);
    free(coo_j);
    free(diag);
    free(ell_values);
    free(ell_cols);
    free(ell_cols64);
    free(x);
    free(y);
    free(y_vectorized);
}

int test_mv_ell_vec_from_openfoam_coo_matrix(char *filename, int N_TESTS)
{

    printf("Loading input data system from file...\n");

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }
    printf("File %s opened successfully.\n", filename);

    int n = 0, count_upper = 0, count_lower = 0;

    double *diag = parseDoubleArray(file, "diag", &n);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *coo_cols = parseIntArray(file, "upperAddr", &count_upper);
    int *coo_rows = parseIntArray(file, "lowerAddr", &count_lower);

    int nnz_max = compute_max_nnz_row_full(n, count_upper, coo_rows, coo_cols);
    printf("nnz_max = %d\n", nnz_max);

    // --- Open file and write CSV header if file is empty ---
    FILE *out = fopen("scripts/data/mv_prof_foam_O3_avg_tail_opt.csv", "a");
    assert(out && "Unable to open output file");

    // Check if file is empty, then write header
    fseek(out, 0, SEEK_END);
    if (ftell(out) == 0)
    {
        fprintf(out, "n,sparsity,max_nnz_row,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
    }

    printf("Running cylinder problem with n cells = %d\n", n);

    double *ell_values = malloc(nnz_max * n * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * n * sizeof(int));
    coo_to_ell_symmetric_full_colmajor(n, count_upper, upper, coo_rows, coo_cols, ell_values, ell_col_idx, nnz_max);
    double sparsity = analyze_ell_matrix_full_colmajor(n, nnz_max, ell_values, ell_col_idx);

    uint64_t *ell_cols64 = malloc(nnz_max * n * sizeof(uint64_t));
    for (int k = 0; k < nnz_max * n; ++k)
        ell_cols64[k] = (uint64_t)ell_col_idx[k];

    double *x = malloc(n * sizeof(double));
    double *y = calloc(n, sizeof(double));
    double *y_vectorized = calloc(n, sizeof(double));

    for (int i = 0; i < n; ++i)
        x[i] = rand() / (double)RAND_MAX;


    struct timespec start, end;
    uint64_t start_cycles, end_cycles;
    double time_serial = 0.0;
    double time_vector = 0.0;
    uint64_t cycles_serial = 0;
    uint64_t cycles_vector = 0;

    // --- Serial ---
    for (int i = 0; i < N_TESTS; i++)
    {
        flush_cache_by_accessing_garbage();
        struct timespec start, end;
        uint64_t start_cycles, end_cycles;
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_values, ell_cols64, x, y);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_serial += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_serial += end_cycles - start_cycles;
    }

    // take average of N_TESTS
    const double cycles_serial_mean = (double)cycles_serial / N_TESTS;
    const double time_serial_mean = time_serial / N_TESTS;

    // --- Vectorized ---
    for (int i = 0; i < N_TESTS; i++)
    {
        flush_cache_by_accessing_garbage();
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        // mv_ell_symmetric_full_colmajor_vector(n, nnz_max, diag, ell_values, ell_cols64, x, y_vectorized);
        mv_ell_symmetric_full_colmajor_vector_vlset_opt(n, nnz_max, diag, ell_values, ell_cols64, x, y_vectorized);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_vector += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_vector += end_cycles - start_cycles;
    }

    // take average on N_TESTS
    const double cycles_vector_mean = (double)cycles_vector / N_TESTS;
    const double time_vector_mean = time_vector / N_TESTS;

    // --- Check result ---
    int pass = 1;
    for (int i = 0; i < n; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6))
        {
            pass = 0;
            break;
        }
    }
    // --- Output ---
    printf("Time serial     : %.6f s\n", time_serial_mean);
    printf("Time vectorized : %.6f s\n", time_vector_mean);
    printf("Speedup (time)  : %.2fx\n", time_serial_mean / time_vector_mean);
    printf("Cycles serial   : %.2f \n", cycles_serial_mean);
    printf("Cycles vector   : %.2f \n", cycles_vector_mean);
    printf("Speedup (cycles): %.2fx\n", cycles_serial_mean / cycles_vector_mean);
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // --- Save to CSV ---
    fprintf(out, "%d,%.4f,%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n, sparsity, nnz_max,
            time_serial_mean, time_vector_mean,
            time_serial_mean / time_vector_mean,
            cycles_serial_mean, cycles_vector_mean,
            cycles_serial_mean / cycles_vector_mean,
            pass ? "PASS" : "FAIL");

    fclose(out);

    // --- Cleanup ---
    fclose(file);
    free(x);
    free(y);
    free(y_vectorized);
    free(diag);
    free(upper);
    free(coo_cols);
    free(coo_rows);
    free(ell_values);
    free(ell_col_idx);
    free(ell_cols64);

    printf("End of program.\n");
    return pass ? 0 : 1;
}

// Uses the official tutorial SAXPY implementation
/*
void saxpy_vec_tutorial(size_t n, const float a, const float *x, float *y) {
  for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
    vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
    vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
    __riscv_vse32_v_f32m8(y, __riscv_vfmacc_vf_f32m8(vy, a, vx, vl), vl);
  }
}
*/
void saxpy_golden(size_t n, double a, double *x, double *y)
{
    for (size_t i = 0; i < n; ++i)
    {
        y[i] = a * x[i] + y[i];
    }
}

void tutorial_saxpy_speedup(size_t n, int N_TESTS)
{
    // --- Open file and write CSV header if file is empty ---
    FILE *out = fopen("scripts/data/saxpy_prof_O3_avg_vlset_opt_perf.csv", "a");
    assert(out && "Unable to open output file");

    // Check if file is empty, then write header
    fseek(out, 0, SEEK_END);
    if (ftell(out) == 0)
    {
        fprintf(out, "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
    }

    printf("Running SAXPY with random arrays of n size = %ld\n", n);

    // generate random data
    // size_t n = 1024 * 1024; // 1 million elements
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double a = 2.0f; // scalar multiplier

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rand() / (double)RAND_MAX;
        y[i] = rand() / (double)RAND_MAX;
    }

    // copy y to to y_vectorized
    double *y_vectorized = malloc(n * sizeof(double));
    memcpy(y_vectorized, y, n * sizeof(double));

    // Measure time for vectorized SAXPY
    struct timespec start, end;
    uint64_t start_cycles, end_cycles;
    double time_scalar = 0.0;
    double time_vector = 0.0;
    uint64_t cycles_scalar = 0;
    uint64_t cycles_vector = 0;

    // Measure time for scalar SAXPY
    for (int i = 0; i < N_TESTS; i++)
    {
        flush_cache_by_accessing_garbage();
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        saxpy_golden(n, a, x, y);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_scalar += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_scalar += end_cycles - start_cycles;
    }

    // take average on N_TESTS
    const double cycles_scalar_mean = (double)cycles_scalar / N_TESTS;
    const double time_scalar_mean = time_scalar / N_TESTS;

    // Measure time for vectorized SAXPY
    for (int i = 0; i < N_TESTS; i++)
    {
        flush_cache_by_accessing_garbage();
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_cycles = read_rdcycle();
        // saxpy_vec_tutorial_double(n, a, x, y_vectorized);
        saxpy_vec_tutorial_double_vlset_opt(n, a, x, y_vectorized);
        end_cycles = read_rdcycle();
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_vector += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        cycles_vector += end_cycles - start_cycles;
    }

    // take average on N_TESTS
    const double cycles_vector_mean = (double)cycles_vector / N_TESTS;
    const double time_vector_mean = time_vector / N_TESTS;

    // --- Check result ---
    int pass = 1;
    for (int i = 0; i < n; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6))
        {
            pass = 0;
            break;
        }
    }

    printf("Scalar SAXPY time: %.6f seconds\n", time_scalar_mean);
    const double speedup = time_scalar_mean / time_vector_mean;
    printf("Speedup: %.2fx\n", speedup);
    printf("Cycles scalar: %.2f\n", cycles_scalar_mean);
    printf("Cycles vectorized: %.2f\n", cycles_vector_mean);
    printf("Speedup (cycles): %.2fx\n", cycles_scalar_mean / cycles_vector_mean);
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // save to CSV
    fprintf(out, "%zu,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s\n",
            n,
            time_scalar_mean, time_vector_mean,
            speedup,
            cycles_scalar_mean, // cycles scalar
            cycles_vector_mean, // cycles vectorized
            (double)(cycles_scalar_mean) / (double)(cycles_vector_mean),
            pass ? "PASS" : "FAIL");

    // Free allocated memory
    free(x);
    free(y);
    free(y_vectorized);
    fclose(out);
}

int main(void)
{
    int N_TESTS = 30;
/*
    // Random mv test
    double sparsity_levels[] = {0.01, 0.02, 0.05, 0.1, 0.2};
    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768};

    for (int i = 0; i < sizeof(sparsity_levels) / sizeof(sparsity_levels[0]); ++i)
    {
        for (int j = 0; j < sizeof(sizes) / sizeof(sizes[0]); ++j)
        {
            mv_rvv_vs_scalar(sizes[j], sparsity_levels[i], N_TESTS);
        }
    }

    // foam mv test
    test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/2000.system", N_TESTS);
    test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/8000.system", N_TESTS);
    test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/32k.system", N_TESTS);
    test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/128k.system", N_TESTS);
*/


    // saxpy test
    int sizes_saxpy[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    printf("%lu \n", sizeof(sizes_saxpy));
    for (int i = 0; i < sizeof(sizes_saxpy) / sizeof(sizes_saxpy[0]); i++)
    {
       tutorial_saxpy_speedup(sizes_saxpy[i], N_TESTS);
    }


    return 0;
}
