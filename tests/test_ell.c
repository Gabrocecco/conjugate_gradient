#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "ell.h"

#define TOL 1e-10

void test_coo_to_ell_with_print()
{
    int n = 5;
    double diag[] = {1, 4, 6, 9, 11};

    // Only upper triangle (j > i), no diagonals
    double values[] = {2, 3, 5, 7, 8, 10};
    int rows[] = {0, 0, 1, 2, 2, 3};
    int cols[] = {1, 4, 2, 3, 4, 4};
    int nnz = 6;

    double expected[5][5] = {
        {1, 2, 0, 0, 3},
        {2, 4, 5, 0, 0},
        {0, 5, 6, 7, 8},
        {0, 0, 7, 9, 10},
        {3, 0, 8, 10, 11}};

    int max_nnz = compute_max_nnz_row_upper(n, nnz, rows, cols);
    assert(max_nnz == 2);

    double *ell_values = calloc(n * max_nnz, sizeof(double));
    int *ell_cols = malloc(n * max_nnz * sizeof(int));

    coo_to_ell_symmetric_upper(n, nnz, values, rows, cols, ell_values, ell_cols, max_nnz);
    // Print ELL values and columns
    printf("ELL Values:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < max_nnz; j++)
        {
            printf("%6.1f ", ell_values[i * max_nnz + j]);
        }
        printf("\n");
    }
    printf("ELL Columns:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < max_nnz; j++)
        {
            printf("%3d ", ell_cols[i * max_nnz + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Convert to dense matrix
    double **A = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        A[i] = calloc(n, sizeof(double));
    convert_ell_to_dense(n, diag, ell_values, ell_cols, max_nnz, A);

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

    mv_ell_symmetric_upper(n, max_nnz, diag, ell_values, ell_cols, x, y);

    for (int i = 0; i < n; i++)
    {
        if (fabs(y[i] - y_expected[i]) > TOL)
        {
            printf("MatVec mismatch at index %d: expected %.2f, got %.2f\n", i, y_expected[i], y[i]);
            assert(0);
        }
    }

    printf("mv_ell_symmetric_upper test passed.\n");

    // Cleanup
    for (int i = 0; i < n; i++)
        free(A[i]);
    free(A);
    free(ell_values);
    free(ell_cols);
}

void test_coo_to_ell_with_print_col_major()
{
    int n = 5;
    double diag[] = {1, 4, 6, 9, 11};

    // Only upper triangle (j > i), no diagonals
    double values[] = {2, 3, 5, 7, 8, 10};
    int rows[] = {0, 0, 1, 2, 2, 3};
    int cols[] = {1, 4, 2, 3, 4, 4};
    int nnz = 6;

    double expected[5][5] = {
        {1, 2, 0, 0, 3},
        {2, 4, 5, 0, 0},
        {0, 5, 6, 7, 8},
        {0, 0, 7, 9, 10},
        {3, 0, 8, 10, 11}};

    int max_nnz = compute_max_nnz_row_upper(n, nnz, rows, cols);
    assert(max_nnz == 2);

    double *ell_values = calloc(n * max_nnz, sizeof(double));
    int *ell_cols = malloc(n * max_nnz * sizeof(int));

    // coo_to_ell_symmetric_upper(n, nnz, values, rows, cols, ell_values, ell_cols, max_nnz);
    coo_to_ell_symmetric_upper_colmajor(n, nnz, values, rows, cols, ell_values, ell_cols, max_nnz);
    // Print ELL values and columns
    printf("ELL Values:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < max_nnz; j++)
        {
            printf("%6.1f ", ell_values[i * max_nnz + j]);
        }
        printf("\n");
    }
    printf("ELL Columns:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < max_nnz; j++)
        {
            printf("%3d ", ell_cols[i * max_nnz + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Convert to dense matrix
    double **A = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        A[i] = calloc(n, sizeof(double));
    // convert_ell_to_dense(n, diag, ell_values, ell_cols, max_nnz, A);
    convert_ell_to_dense_colmajor(n, diag, ell_values, ell_cols, max_nnz, A);

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

    // mv_ell_symmetric_upper(n, max_nnz, diag, ell_values, ell_cols, x, y);
    mv_ell_symmetric_upper_colmajor(n, max_nnz, diag, ell_values, ell_cols, x, y);

    for (int i = 0; i < n; i++)
    {
        if (fabs(y[i] - y_expected[i]) > TOL)
        {
            printf("MatVec mismatch at index %d: expected %.2f, got %.2f\n", i, y_expected[i], y[i]);
            assert(0);
        }
    }

    printf("mv_ell_symmetric_upper test passed.\n");

    // Cleanup
    for (int i = 0; i < n; i++)
        free(A[i]);
    free(A);
    free(ell_values);
    free(ell_cols);
}

void test_coo_to_ell_full_coll_major()
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
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < max_nnz; ++j) {
            int idx = j * n + i;
            printf("%6.1f ", ell_values[idx]);
        }
        printf("\n");
    }
    printf("\nELL cols (row × slot):\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < max_nnz; ++j) {
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

    // mv_ell_symmetric_upper(n, max_nnz, diag, ell_values, ell_cols, x, y);
    int max_nnz_row = compute_max_nnz_row_full(n, nnz, upper_rows, upper_cols);
    // mv_ell_symmetric_upper_colmajor(n, max_nnz, diag, ell_values, ell_cols, x, y);
    mv_ell_symmetric_full_colmajor(n, max_nnz_row, diag, ell_values, ell_cols, x, y);

    for (int i = 0; i < n; i++)
    {
        if (fabs(y[i] - y_expected[i]) > TOL)
        {
            printf("MatVec mismatch at index %d: expected %.2f, got %.2f\n", i, y_expected[i], y[i]);
            assert(0);
        }
    }

    printf("mv_ell_symmetric_upper test passed.\n");

    for (int i = 0; i < n; i++)
        free(A[i]);

    free(ell_values);
    free(ell_cols);
    free(A); 
}

int main()
{
    // test_coo_to_ell_with_print();
    // test_coo_to_ell_with_print();
    // test_coo_to_ell_with_print_col_major();

    test_coo_to_ell_full_coll_major();

    return 0;
}
