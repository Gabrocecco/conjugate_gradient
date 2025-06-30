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

void mv_rvv_vs_scalar(int n, double sparsity)
{
    // --- Open file and write CSV header if file is empty ---
    FILE *out = fopen("scripts/data/mv_prof_random.csv", "a");
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

    // --- Serial ---
    struct timespec start, end;
    uint64_t start_cycles, end_cycles;
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    mv_ell_symmetric_full_colmajor_sdtint(n, max_nnz, diag, ell_values, ell_cols64, x, y);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_serial = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    uint64_t cycles_serial = end_cycles - start_cycles;

    // --- Vectorized ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    mv_ell_symmetric_full_colmajor_vector(n, max_nnz, diag, ell_values, ell_cols64, x, y_vectorized);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_vector = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    uint64_t cycles_vector = end_cycles - start_cycles;

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
    printf("Time serial     : %.6f s\n", time_serial);
    printf("Time vectorized : %.6f s\n", time_vector);
    printf("Speedup (time)  : %.2fx\n", time_serial / time_vector);
    printf("Cycles serial   : %" PRIu64 "\n", cycles_serial);
    printf("Cycles vector   : %" PRIu64 "\n", cycles_vector);
    printf("Speedup (cycles): %.2fx\n", (double)cycles_serial / (double)cycles_vector);
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // --- Save to CSV ---
    fprintf(out, "%d,%.4f,%d,%.6f,%.6f,%.2f,%" PRIu64 ",%" PRIu64 ",%.2f,%s\n",
            n, sparsity, max_nnz,
            time_serial, time_vector,
            time_serial / time_vector,
            cycles_serial, cycles_vector,
            (double)cycles_serial / (double)cycles_vector,
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

int test_mv_ell_vec_from_openfoam_coo_matrix(char *filename)
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
    FILE *out = fopen("scripts/data/mv_prof_foam.csv", "a");
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

    // --- Serial ---
    struct timespec start, end;
    uint64_t start_cycles, end_cycles;
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_values, ell_cols64, x, y);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_serial = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    uint64_t cycles_serial = end_cycles - start_cycles;

    // --- Vectorized ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    mv_ell_symmetric_full_colmajor_vector(n, nnz_max, diag, ell_values, ell_cols64, x, y_vectorized);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_vector = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    uint64_t cycles_vector = end_cycles - start_cycles;

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
    printf("Time serial     : %.6f s\n", time_serial);
    printf("Time vectorized : %.6f s\n", time_vector);
    printf("Speedup (time)  : %.2fx\n", time_serial / time_vector);
    printf("Cycles serial   : %" PRIu64 "\n", cycles_serial);
    printf("Cycles vector   : %" PRIu64 "\n", cycles_vector);
    printf("Speedup (cycles): %.2fx\n", (double)cycles_serial / (double)cycles_vector);
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // --- Save to CSV ---
    fprintf(out, "%d,%.4f,%d,%.6f,%.6f,%.2f,%" PRIu64 ",%" PRIu64 ",%.2f,%s\n",
            n, sparsity, nnz_max,
            time_serial, time_vector,
            time_serial / time_vector,
            cycles_serial, cycles_vector,
            (double)cycles_serial / (double)cycles_vector,
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
void saxpy_golden(size_t n, const float a, const float *x, float *y)
{
    for (size_t i = 0; i < n; ++i)
    {
        y[i] = a * x[i] + y[i];
    }
}

int tutorial_saxpy_speedup(size_t n)
{
    // --- Open file and write CSV header if file is empty ---
    FILE *out = fopen("scripts/data/saxpy_prof.csv", "a");
    assert(out && "Unable to open output file");

    // Check if file is empty, then write header
    fseek(out, 0, SEEK_END);
    if (ftell(out) == 0)
    {
        fprintf(out, "n,time_serial,time_vectorized,speedup_time,cycles_serial,cycles_vector,speedup_cycles,pass\n");
    }

    printf("Running SAXPY with random arrays of n size = %d\n", n);

    // generate random data
    // size_t n = 1024 * 1024; // 1 million elements
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float a = 2.0f; // scalar multiplier

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = rand() / (float)RAND_MAX;
        y[i] = rand() / (float)RAND_MAX;
    }

    // copy y to to y_vectorized
    float *y_vectorized = malloc(n * sizeof(float));
    memcpy(y_vectorized, y, n * sizeof(double));

    // Measure time for vectorized SAXPY
    struct timespec start, end;
    uint64_t start_cycles, end_cycles;

    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    saxpy_vec_tutorial(n, a, x, y_vectorized);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_vectorized = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    printf("Vectorized SAXPY time: %.6f seconds\n", time_vectorized);

    // Measure time for scalar SAXPY
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_cycles = read_rdcycle();
    saxpy_golden(n, a, x, y);
    end_cycles = read_rdcycle();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_scalar = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

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

    printf("Scalar SAXPY time: %.6f seconds\n", time_scalar);
    double speedup = time_scalar / time_vectorized;
    printf("Speedup: %.2fx\n", speedup);
    printf("Cycles scalar: %" PRIu64 "\n", end_cycles - start_cycles);
    printf("Cycles vectorized: %" PRIu64 "\n", end_cycles - start_cycles);
    printf("Speedup (cycles): %.2fx\n", (double)(end_cycles - start_cycles) / (double)(end_cycles - start_cycles));
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // save to CSV
    fprintf(out, "%zu,%.6f,%.6f,%.2f,%" PRIu64 ",%" PRIu64 ",%.2f,%s\n",
            n,
            time_scalar, time_vectorized,
            speedup,
            end_cycles - start_cycles, // cycles scalar
            end_cycles - start_cycles, // cycles vectorized
            (double)(end_cycles - start_cycles) / (double)(end_cycles - start_cycles),
            pass ? "PASS" : "FAIL");

    // Free allocated memory
    free(x);
    free(y);
    free(y_vectorized);
    fclose(out);
}

int main(void)
{
    // double sparsity_levels[] = {0.01, 0.02, 0.05, 0.1, 0.2};
    // int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768};

    // for (int i = 0; i < sizeof(sparsity_levels) / sizeof(sparsity_levels[0]); ++i)
    // {
    //     for (int j = 0; j < sizeof(sizes) / sizeof(sizes[0]); ++j)
    //     {
    //         mv_rvv_vs_scalar(sizes[j], sparsity_levels[i]);
    //     }
    // }

    // test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/2000.system");
    // test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/8000.system");
    // test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/32k.system");
    // test_mv_ell_vec_from_openfoam_coo_matrix("data/cylinder/128k.system");

    tutorial_saxpy_speedup(1024 * 1024);     // 1 million elements
    tutorial_saxpy_speedup(1024 * 1024 * 8); // 8 million elements

    return 0;
}
