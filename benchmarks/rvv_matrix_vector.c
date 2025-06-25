#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <assert.h>
#include <time.h>
#include "common.h"
#include "vectorized.h"
#include "ell.h"
#include "coo.h"
#include "csr.h"
#include "parser.h"
#include "mmio.h"

void mv_rvv_vs_scalar(int n, double sparsity)
{
    printf("Running with n = %d, sparsity = %.2f\n", n, sparsity);

    // 1) Generate random COO (upper triangle + diagonal)
    int upper_nnz;
    double *coo_vals, *diag;
    int *coo_i, *coo_j;
    generate_sparse_symmetric_coo(n, sparsity,
                                  &upper_nnz,
                                  &coo_vals,
                                  &coo_i,
                                  &coo_j,
                                  &diag);

    printf("n = %d, upper_nnz = %d\n", n, upper_nnz);
    // print_dense_symmeric_matrix_from_coo(n, diag,
    //                                      coo_vals, coo_i, coo_j, upper_nnz);

    // 2) Compute maximum non-zeros per row for full (symmetric) matrix
    int max_nnz = compute_max_nnz_row_full(n, upper_nnz, coo_i, coo_j);
    printf("max_nnz_row = %d\n", max_nnz);

    // 3) Allocate and fill ELL storage (column-major)
    double *ell_values = calloc((size_t)n * max_nnz, sizeof(*ell_values));
    int *ell_cols = malloc((size_t)n * max_nnz * sizeof(*ell_cols));
    coo_to_ell_symmetric_full_colmajor(
        n, upper_nnz,
        coo_vals, coo_i, coo_j,
        ell_values, ell_cols,
        max_nnz);
    // analyze_ell_matrix_full_colmajor(n, max_nnz, ell_values, ell_cols);

    // 4) Allocate vectors x, y (serial), y_vectorized
    double *x = malloc((size_t)n * sizeof(*x));
    double *y = calloc((size_t)n, sizeof(*y));
    double *y_vectorized = calloc((size_t)n, sizeof(*y_vectorized));

    // 5) Fill x with random values in [0,1)
    for (int i = 0; i < n; ++i)
    {
        x[i] = rand() / (double)RAND_MAX;
    }

    // 6) Convert ell_cols to 64-bit indices if required
    uint64_t *ell_cols64 = malloc((size_t)n * max_nnz * sizeof(*ell_cols64));
    for (int k = 0; k < n * max_nnz; ++k)
    {
        ell_cols64[k] = (uint64_t)ell_cols[k];
    }

    // 7) Perform mat-vec: serial and vectorized
    struct timespec start, end;

    // Serial
    clock_gettime(CLOCK_MONOTONIC, &start);
    mv_ell_symmetric_full_colmajor_sdtint(
        n, max_nnz,
        diag, ell_values, ell_cols64,
        x, y);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_serial = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    // Vectorized
    clock_gettime(CLOCK_MONOTONIC, &start);
    mv_ell_symmetric_full_colmajor_vector(
        n, max_nnz,
        diag, ell_values, ell_cols64,
        x, y_vectorized);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_vector = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    printf("Time serial     : %.6f s\n", time_serial);
    printf("Time vectorized : %.6f s\n", time_vector);
    printf("Speedup         : %.2fx\n", time_serial / time_vector);

    // 8) Print results side by side
    // puts(" y (serial)   |  y (vectorized)");
    // for (int i = 0; i < n; ++i) {
    //     printf("%12g  |  %12g\n", y[i], y_vectorized[i]);
    // }

    // 9) Verify equality within tolerance
    int pass = 1;
    for (int i = 0; i < n; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6))
        {
            pass = 0;
            break;
        }
    }
    printf("%s\n\n", pass ? "PASS: Results match!"
                          : "FAIL: Results do NOT match!");

    // 10) Free all allocated memory
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
    if (!file) {
        perror("Error in opening file");
        return -1;
    }
    printf("File %s opened successfully.\n", filename);

    int n = 0, count_upper = 0, count_lower = 0;

    double *diag     = parseDoubleArray(file, "diag", &n);
    double *upper    = parseDoubleArray(file, "upper", &count_upper);
    int *coo_cols    = parseIntArray(file, "upperAddr", &count_upper);
    int *coo_rows    = parseIntArray(file, "lowerAddr", &count_lower);

    int nnz_max = compute_max_nnz_row_full(n, count_upper, coo_rows, coo_cols);
    printf("nnz_max = %d\n", nnz_max);

    double *ell_values = malloc(nnz_max * n * sizeof(double));
    int *ell_col_idx   = malloc(nnz_max * n * sizeof(int));
    coo_to_ell_symmetric_full_colmajor(n, count_upper, upper, coo_rows, coo_cols, ell_values, ell_col_idx, nnz_max);
    analyze_ell_matrix_full_colmajor(n, nnz_max, ell_values, ell_col_idx);

    uint64_t *ell_cols64 = malloc(nnz_max * n * sizeof(uint64_t));
    for (int k = 0; k < nnz_max * n; ++k)
        ell_cols64[k] = (uint64_t)ell_col_idx[k];

    double *x             = malloc(n * sizeof(double));
    double *y             = calloc(n, sizeof(double));
    double *y_vectorized  = calloc(n, sizeof(double));

    for (int i = 0; i < n; ++i)
        x[i] = rand() / (double)RAND_MAX;

    struct timespec t0, t1;

    // --- Serial ---
    clock_gettime(CLOCK_MONOTONIC, &t0);

    mv_ell_symmetric_full_colmajor_sdtint(n, nnz_max, diag, ell_values, ell_cols64, x, y);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double time_serial = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    // --- Vectorized ---
    clock_gettime(CLOCK_MONOTONIC, &t0);

    mv_ell_symmetric_full_colmajor_vector(n, nnz_max, diag, ell_values, ell_cols64, x, y_vectorized);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double time_vector = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    // --- Output ---
    printf("Time serial     : %.6f s\n", time_serial);
    printf("Time vectorized : %.6f s\n", time_vector);
    printf("Speedup (time)  : %.2fx\n", time_serial / time_vector);

    // --- Check result ---
    int pass = 1;
    for (int i = 0; i < n; ++i) {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6)) {
            pass = 0;
            break;
        }
    }
    printf("%s\n\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

    // --- Cleanup ---
    fclose(file);
    free(x); free(y); free(y_vectorized);
    free(diag); free(upper);
    free(coo_cols); free(coo_rows);
    free(ell_values); free(ell_col_idx); free(ell_cols64);

    printf("End of program.\n");
    return pass ? 0 : 1;
}

int main(void)
{
    //int n = 32768;
    //double sparsity = 0.01;
    //mv_rvv_vs_scalar(n, sparsity);


    test_mv_ell_vec_from_openfoam_coo_matrix("data/matrix_128k_mm_upper.txt");
    return 0;
}
