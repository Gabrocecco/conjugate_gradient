#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <assert.h>
#include "common.h"
#include "vectorized.h"
#include "ell.h"

/*
riscv64-unknown-elf-gcc -O0 -march=rv64gcv -mabi=lp64d   -std=c99 -Wall -pedantic -Iinclude   -o build/test_vec src/vectorized.c src/ell.c tests/test_vec.c

spike --isa=rv64gcv pk build/test_vec

*/

#define TOL 1e-10

void test_mv_ell_vec_with_5x5_matrix()
{
    #define N 5
    #define MAX_NNZ 3

    // Diagonal
    double diag[N] = {10.0, 20.0, 30.0, 40.0, 50.0};

    double ell_values[MAX_NNZ * N] = {
        /* slot0 */ 1.0, 1.0, 3.0, 2.0, 5.0,
        /* slot1 */ 2.0, 3.0, 4.0, 4.0, 2.0,
        /* slot2 */ 0.0, 0.0, 0.0, 1.0, 0.0};

    uint64_t ell_cols[MAX_NNZ * N] = {
        /* slot0 */ 1, 0, 1, 0, 1,
        /* slot1 */ 3, 2, 3, 2, 3,
        /* slot2 */ -1, -1, -1, 4, -1};

    double x[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[N];
    double y_vectorized[N];

    // Stampa matrice completa
    print_dense_matrix_from_ell(N, MAX_NNZ, diag, ell_values, ell_cols);

    // Stampa contenuto ELL
    print_ell_format(N, MAX_NNZ, ell_values, ell_cols);

    // Esegui prodotto y = A * x
    mv_ell_symmetric_full_colmajor_sdtint(
        N, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y);

    printf("Result y = A * x:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("  y[%d] = %g\n", i, y[i]);
    }

    mv_ell_symmetric_full_colmajor_vector(
        N, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y_vectorized);

    printf("Result y (vectorized) = A * x:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("  y[%d] = %g\n", i, y_vectorized[i]);
    }

    // Check if results match
    int pass = 1;
    for (int i = 0; i < N; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6f))
        {
            printf("FAIL: y[%d] = %g != %g (vectorized)\n", i, y[i], y_vectorized[i]);
            pass = 0;
        }
    }
    if (pass)
    {
        printf("PASS: Results match!\n");
    }
    else
    {
        printf("FAIL: Results do not match!\n");
    }
}

void test_mv_ell_vec_from_coo_matrix()
{
    int n = 5;
    double diag[] = {1, 4, 6, 9, 11};

    // Only upper triangle (j > i), no diagonals
    double upper_values[] = {2, 3, 5, 7, 8, 10};
    int upper_rows[] = {0, 0, 1, 2, 2, 3};
    int upper_cols[] = {1, 4, 2, 3, 4, 4};
    int nnz = 6;

    double expected[5][5] = {
        {1, 2, 0, 0, 3},
        {2, 4, 5, 0, 0},
        {0, 5, 6, 7, 8},
        {0, 0, 7, 9, 10},
        {3, 0, 8, 10, 11}};

    int max_nnz = compute_max_nnz_row_full(n, nnz, upper_rows, upper_cols);
    printf("max_nnz = %d \n", max_nnz);

    double *ell_values = calloc(n * max_nnz, sizeof(double));
    int *ell_cols = malloc(n * max_nnz * sizeof(int));

    coo_to_ell_symmetric_full_colmajor(n, nnz, upper_values, upper_rows, upper_cols, ell_values, ell_cols, max_nnz);

    // Print ELL values and columns
    printf("ELL values (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < max_nnz; ++j)
        {
            int idx = j * n + i;
            printf("%6.1f ", ell_values[idx]);
        }
        printf("\n");
    }
    printf("\nELL cols (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < max_nnz; ++j)
        {
            int idx = j * n + i;
            printf("%3d ", ell_cols[idx]);
        }
        printf("\n");
    }
    printf("\n");

    // Convert to dense matrix
    double **A = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        A[i] = calloc(n, sizeof(double));

    convert_ell_full_to_dense_colmajor(n, diag, ell_values, ell_cols, max_nnz, A);

    // Check against expected
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            assert(fabs(A[i][j] - expected[i][j]) < TOL);

    printf("test_coo_to_ell_with_print passed.\n");
    print_dense_matrix(A, n);

    // === MatVec test ===
    double x[] = {1, 2, 3, 4, 5};
    double y[5] = {0};
    double y_expected[] = {20, 25, 96, 107, 122};
    double y_vectorized[5];

    int max_nnz_row = compute_max_nnz_row_full(n, nnz, upper_rows, upper_cols);

    uint64_t *ell_cols64 = malloc(max_nnz_row * n * sizeof(uint64_t));
    for (int k = 0; k < max_nnz_row * n; ++k)
    {
        ell_cols64[k] = (uint64_t)ell_cols[k];
    }

    mv_ell_symmetric_full_colmajor_sdtint(n, max_nnz_row, diag, ell_values, ell_cols64, x, y);

    for (int i = 0; i < n; i++)
    {
        if (fabs(y[i] - y_expected[i]) > TOL)
        {
            printf("MatVec mismatch at index %d: expected %.2f, got %.2f\n", i, y_expected[i], y[i]);
            assert(0);
        }
    }

    // mv_ell_symmetric_full_colmajor_vector(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    // mv_ell_symmetric_full_colmajor_vector_m2(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    // mv_ell_symmetric_full_colmajor_vector_debug(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    mv_ell_symmetric_full_colmajor_vector_m8(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    printf("Result y (vectorized) = A * x:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("  y[%d] = %g\n", i, y_vectorized[i]);
    }

    // Check if results match
    int pass = 1;
    for (int i = 0; i < N; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6f))
        {
            printf("FAIL: y[%d] = %g != %g (vectorized)\n", i, y[i], y_vectorized[i]);
            pass = 0;
        }
    }
    if (pass)
    {
        printf("PASS: Results match!\n");
    }
    else
    {
        printf("FAIL: Results do not match!\n");
    }

    printf("mv_ell_symmetric_upper test passed.\n");

    for (int i = 0; i < n; i++)
        free(A[i]);

    free(ell_values);
    free(ell_cols);
    free(A);
}

// void test_mv_ell_vec_drom_random_coo_matrix(){

// }

int main(void)
{
    // test_mv_ell_vec_with_5x5_matrix();

    test_mv_ell_vec_from_coo_matrix();

    // test_mv_ell_vec_drom_random_coo_matrix();

    return 0;
}