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

/*
riscv64-unknown-elf-gcc -O0 -march=rv64gcv -mabi=lp64d   -std=c99 -Wall -pedantic -Iinclude   -o build/test_vec src/vectorized.c src/ell.c tests/test_vec.c

spike --isa=rv64gcv pk build/test_vec

*/

#define TOL 1e-10

void test_mv_ell_vec_with_5x5_matrix()
{
#define N_DIM 5
#define MAX_NNZ 3

    // Diagonal
    double diag[N_DIM] = {10.0, 20.0, 30.0, 40.0, 50.0};

    double ell_values[MAX_NNZ * N_DIM] = {
        /* slot0 */ 1.0, 1.0, 3.0, 2.0, 5.0,
        /* slot1 */ 2.0, 3.0, 4.0, 4.0, 2.0,
        /* slot2 */ 0.0, 0.0, 0.0, 1.0, 0.0};

    uint64_t ell_cols[MAX_NNZ * N_DIM] = {
        /* slot0 */ 1, 0, 1, 0, 1,
        /* slot1 */ 3, 2, 3, 2, 3,
        /* slot2 */ -1, -1, -1, 4, -1};

    double x[N_DIM] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[N_DIM];
    double y_vectorized[N_DIM];

    // Stampa matrice completa
    print_dense_matrix_from_ell(N_DIM, MAX_NNZ, diag, ell_values, ell_cols);

    // Stampa contenuto ELL
    print_ell_format(N_DIM, MAX_NNZ, ell_values, ell_cols);

    // Esegui prodotto y = A * x
    mv_ell_symmetric_full_colmajor_sdtint(
        N_DIM, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y);

    printf("Result y = A * x:\n");
    for (int i = 0; i < N_DIM; ++i)
    {
        printf("  y[%d] = %g\n", i, y[i]);
    }

    mv_ell_symmetric_full_colmajor_vector_debug(
        N_DIM, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y_vectorized);

    printf("Result y (vectorized) = A * x:\n");
    for (int i = 0; i < N_DIM; ++i)
    {
        printf("  y[%d] = %g\n", i, y_vectorized[i]);
    }

    // Check if results match
    int pass = 1;
    for (int i = 0; i < N_DIM; ++i)
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
    printf("ELL values (row x slot):\n");
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
    mv_ell_symmetric_full_colmajor_vector_debug(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    // mv_ell_symmetric_full_colmajor_vector_m8(n, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);
    printf("Result y (vectorized) = A * x:\n");
    for (int i = 0; i < N_DIM; ++i)
    {
        printf("  y[%d] = %g\n", i, y_vectorized[i]);
    }

    // Check if results match
    int pass = 1;
    for (int i = 0; i < N_DIM; ++i)
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

/*
 * Test mat-vec multiply (serial vs. vectorized) on a random sparse symmetric matrix.
 * n         : matrix dimension
 * sparsity  : probability of a non-zero in each upper-triangular position
 */
void test_mv_ell_vec_from_random_coo_matrix(int n, double sparsity)
{
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
    mv_ell_symmetric_full_colmajor_sdtint(
        n, max_nnz,
        diag, ell_values, ell_cols64,
        x, y);
    mv_ell_symmetric_full_colmajor_vector(
        n, max_nnz,
        diag, ell_values, ell_cols64,
        x, y_vectorized);

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
    // const char *filename = "data/data.txt"; // linear system data input (COO) filename
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filename);
    }

    int n = 0, count_upper = 0, count_lower = 0, count_solution_openfoam = 0;

    // Parse all sections in a single file pass
    double *diag = parseDoubleArray(file, "diag", &n);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *coo_cols = parseIntArray(file, "upperAddr", &count_upper);
    int *coo_rows = parseIntArray(file, "lowerAddr", &count_lower);

    int nnz_max = compute_max_nnz_row_full(n, count_upper, coo_rows, coo_cols);
    printf("nnz_max = %d\n", nnz_max);

    // Allocate memory for the ELL format
    double *ell_values = malloc(nnz_max * n * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * n * sizeof(int));

    // Convert the COO format to ELL format
    coo_to_ell_symmetric_full_colmajor(n, count_upper, upper, coo_rows, coo_cols, ell_values, ell_col_idx, nnz_max);

    // Print the ELL format analysis
    analyze_ell_matrix_full_colmajor(n, nnz_max, ell_values, ell_col_idx);

    uint64_t *ell_cols64 = malloc(nnz_max * n * sizeof(uint64_t));
    for (int k = 0; k < nnz_max * n; ++k)
    {
        ell_cols64[k] = (uint64_t)ell_col_idx[k];
    }

    double *x = malloc((size_t)n * sizeof(*x));
    double *y = calloc((size_t)n, sizeof(*y));
    double *y_vectorized = calloc((size_t)n, sizeof(*y_vectorized));

    // 5) Fill x with random values in [0,1)
    for (int i = 0; i < n; ++i)
    {
        x[i] = rand() / (double)RAND_MAX;
    }
    mv_ell_symmetric_full_colmajor_sdtint(
        n, nnz_max,
        diag, ell_values, ell_cols64,
        x, y);
    mv_ell_symmetric_full_colmajor_vector(
        n, nnz_max,
        diag, ell_values, ell_cols64,
        x, y_vectorized);

    // 8) Print results side by side
    puts(" y (serial)   |  y (vectorized)");
    for (int i = 0; i < n; ++i)
    {
        printf("%12g  |  %12g\n", y[i], y_vectorized[i]);
    }

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

    // Free allocated memory
    free(x);
    free(y);
    free(y_vectorized);
    free(diag);
    free(upper);
    free(coo_cols);
    free(coo_rows);
    free(ell_values);
    free(ell_col_idx);

    fclose(file);
    printf("Files closed successfully.\n");
    printf("End of program.\n");
}

/*
 * Read a symmetric matrix in Matrix-Market COO format, convert it to
 * ELL (full symmetric, column-major), then compare serial vs vectorized
 * matrix-vector products.
 *
 * filename : path to the Matrix Market file
 */
void test_mv_ell_vec_from_matrix_market(const char *filename)
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int *I, *J;
    double *val;

    // 1) open and read banner
    if ((f = fopen(filename, "r")) == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }
    // we only support real, sparse, symmetric or general square
    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
    {
        fprintf(stderr, "Only sparse real matrices supported.\n");
        exit(EXIT_FAILURE);
    }
    // 2) read dimensions
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
    {
        fprintf(stderr, "Failed to read matrix size.\n");
        exit(EXIT_FAILURE);
    }
    if (M != N)
    {
        fprintf(stderr, "Matrix must be square (M=%d, N=%d)\n", M, N);
        exit(EXIT_FAILURE);
    }

    // 3) allocate for COO input
    I = malloc(nz * sizeof(*I));
    J = malloc(nz * sizeof(*J));
    val = malloc(nz * sizeof(*val));
    if (!I || !J || !val)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // 4) read all entries (1-based to 0-based)
    for (int k = 0; k < nz; ++k)
    {
        int ii, jj;
        double vv;
        fscanf(f, "%d %d %lg\n", &ii, &jj, &vv);
        I[k] = ii - 1;
        J[k] = jj - 1;
        val[k] = vv;
    }
    fclose(f);

    // 5) split into diagonal + strictly-upper arrays
    double *diag = calloc(N, sizeof(*diag));
    int upper_nnz = 0;
    // first pass: count strict-upper entries
    for (int k = 0; k < nz; ++k)
    {
        int i = I[k], j = J[k];
        if (i == j)
        {
            diag[i] = val[k];
        }
        else
        {
            // treat every off-diag as one strict upper entry
            if (i < j)
            {
                ++upper_nnz;
            }
            else if (mm_is_symmetric(matcode))
            {
                // symmetric file: entry (j,i) implied, count (j,i)
                ++upper_nnz;
            }
        }
    }

    // 6) allocate strict-upper COO arrays
    double *coo_vals_upper = malloc(upper_nnz * sizeof(*coo_vals_upper));
    int *coo_rows = malloc(upper_nnz * sizeof(*coo_rows));
    int *coo_cols = malloc(upper_nnz * sizeof(*coo_cols));
    if (!coo_vals_upper || !coo_rows || !coo_cols)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    // second pass: fill them
    int idx = 0;
    for (int k = 0; k < nz; ++k)
    {
        int i = I[k], j = J[k];
        double v = val[k];
        if (i == j)
            continue;
        if (i < j)
        {
            coo_rows[idx] = i;
            coo_cols[idx] = j;
            coo_vals_upper[idx] = v;
            ++idx;
        }
        else if (mm_is_symmetric(matcode))
        {
            // flip into strict upper
            coo_rows[idx] = j;
            coo_cols[idx] = i;
            coo_vals_upper[idx] = v;
            ++idx;
        }
    }

    printf("Read %d×%d matrix, strict upper nnz = %d\n", N, N, upper_nnz);

    // 7) print full dense for verification
    // print_dense_symmeric_matrix_from_coo(
    //     N, diag,
    //     coo_vals_upper, coo_rows, coo_cols,
    //     upper_nnz
    // );

    // 8) build ELL
    int max_nnz_row = compute_max_nnz_row_full(
        N, upper_nnz, coo_rows, coo_cols);
    printf("max_nnz_row = %d\n", max_nnz_row);
    double *ell_values = calloc((size_t)N * max_nnz_row, sizeof(*ell_values));
    int *ell_cols = malloc((size_t)N * max_nnz_row * sizeof(*ell_cols));
    coo_to_ell_symmetric_full_colmajor(
        N, upper_nnz,
        coo_vals_upper, coo_rows, coo_cols,
        ell_values, ell_cols,
        max_nnz_row);
    analyze_ell_matrix_full_colmajor(N, max_nnz_row, ell_values, ell_cols);

    // 9) allocate and fill random x
    double *x = malloc((size_t)N * sizeof(*x));
    for (int i = 0; i < N; ++i)
        x[i] = rand() / (double)RAND_MAX;

    // 10) allocate y and y_vectorized
    double *y = calloc((size_t)N, sizeof(*y));
    double *y_vectorized = calloc((size_t)N, sizeof(*y_vectorized));

    // convert cols to 64-bit
    uint64_t *ell_cols64 = malloc((size_t)N * max_nnz_row * sizeof(*ell_cols64));
    for (int k = 0; k < N * max_nnz_row; ++k)
        ell_cols64[k] = (uint64_t)ell_cols[k];

    // 11) run mat-vec: serial and vectorized
    mv_ell_symmetric_full_colmajor_sdtint(
        N, max_nnz_row, diag, ell_values, ell_cols64, x, y);
    mv_ell_symmetric_full_colmajor_vector(
        N, max_nnz_row, diag, ell_values, ell_cols64, x, y_vectorized);

    // 12) print and compare
    puts(" y (serial)   |  y (vectorized)");
    for (int i = 0; i < N; ++i)
    {
        printf("%12g  |  %12g\n", y[i], y_vectorized[i]);
    }
    int pass = 1;
    for (int i = 0; i < N; ++i)
        if (!fp_eq(y[i], y_vectorized[i], 1e-6))
            pass = 0;
    printf("%s\n", pass ? "PASS: match!" : "FAIL: mismatch!");

    // 13) clean up
    free(I);
    free(J);
    free(val);
    free(diag);
    free(coo_vals_upper);
    free(coo_rows);
    free(coo_cols);
    free(ell_values);
    free(ell_cols);
    free(ell_cols64);
    free(x);
    free(y);
    free(y_vectorized);
}

void test_dot_product_vec()
{
    printf("=== dot product test ===\n");

    // --- Fixed small test ---
    {
        double a[] = {1.0, 2.0, 3.0, 4.0};
        double b[] = {5.0, 6.0, 7.0, 8.0};
        int n = sizeof(a) / sizeof(a[0]);

        // compute reference
        double expected = 0.0;
        for (int i = 0; i < n; ++i)
        {
            expected += a[i] * b[i];
        }

        // compute with vectorized dot
        double result = vec_dot_vectorized_debug(a, b, n);

        printf("Fixed test (n=%d): expected = %g, vec_dot = %g -> %s\n",
               n, expected, result,
               fp_eq(expected, result, 1e-6) ? "PASS" : "FAIL");
    }

    // --- Randomized test ---
    {
        const int n = 1024;
        double *a = malloc(n * sizeof(*a));
        double *b = malloc(n * sizeof(*b));
        if (!a || !b)
        {
            perror("malloc");
            exit(EXIT_FAILURE);
        }

        // fill with random [0,1)
        for (int i = 0; i < n; ++i)
        {
            a[i] = rand() / (double)RAND_MAX;
            b[i] = rand() / (double)RAND_MAX;
        }

        // scalar reference
        double expected = 0.0;
        for (int i = 0; i < n; ++i)
        {
            expected += a[i] * b[i];
        }

        // vectorized
        double result = vec_dot_vectorized(a, b, n);

        printf("Random test (n=%d): expected = %g, vec_dot = %g -> %s\n",
               n, expected, result,
               fp_eq(expected, result, 1e-6) ? "PASS" : "FAIL");

        free(a);
        free(b);
    }

    printf("\n");
}

void test_vec_axpy_vector()
{
    printf("=== AXPY vectorized test ===\n");

    // ---- Fixed small test ----
    {
        double a[] = {1.0, 2.0, 3.0, 4.0};
        double b[] = {5.0, 6.0, 7.0, 8.0};
        double alpha = 2.0;
        int n = sizeof(a) / sizeof(a[0]);
        double out_ref[4], out_vec[4];

        for (int i = 0; i < n; ++i)
            out_ref[i] = a[i] + alpha * b[i];
        vec_axpy_vectorized_debug(a, b, alpha, out_vec, n);

        int pass = 1;
        for (int i = 0; i < n; ++i)
        {
            if (!fp_eq(out_ref[i], out_vec[i], 1e-9))
            {
                pass = 0;
                printf("FAIL idx=%d: ref=%g vec=%g\n",
                       i, out_ref[i], out_vec[i]);
            }
        }
        printf("Fixed test (n=%d): %s\n\n", n,
               pass ? "PASS" : "FAIL");
    }

    // ---- Randomized test ----
    {
        const int n = 1024;
        double *a = malloc(n * sizeof(*a));
        double *b = malloc(n * sizeof(*b));
        double *out_ref = malloc(n * sizeof(*out_ref));
        double *out_vec = malloc(n * sizeof(*out_vec));
        double alpha = 0.37;

        // fill with random [0,1)
        for (int i = 0; i < n; ++i)
        {
            a[i] = rand() / (double)RAND_MAX;
            b[i] = rand() / (double)RAND_MAX;
        }

        for (int i = 0; i < n; ++i)
            out_ref[i] = a[i] + alpha * b[i];
        vec_axpy_vectorized(a, b, alpha, out_vec, n);

        int pass = 1;
        for (int i = 0; i < n; ++i)
        {
            if (!fp_eq(out_ref[i], out_vec[i], 1e-9))
            {
                pass = 0;
                printf("FAIL idx=%d: ref=%g vec=%g\n",
                       i, out_ref[i], out_vec[i]);
                break;
            }
        }
        printf("Random test (n=%d): %s\n\n", n,
               pass ? "PASS" : "FAIL");

        free(a);
        free(b);
        free(out_ref);
        free(out_vec);
    }
}

int main(void)
{
    // test_mv_ell_vec_with_5x5_matrix();

    // test_mv_ell_vec_from_coo_matrix();

    // Seed the random number generator once
    srand((unsigned)time(NULL));

    // Run tests for different sizes/sparsities
    // test_mv_ell_vec_from_random_coo_matrix(5, 0.30);
    // test_mv_ell_vec_from_random_coo_matrix(10, 0.10);
    // test_mv_ell_vec_from_random_coo_matrix(50, 0.02);
    // test_mv_ell_vec_from_random_coo_matrix(100, 0.30);
    // test_mv_ell_vec_from_random_coo_matrix(150, 0.10);
    // test_mv_ell_vec_from_random_coo_matrix(200, 0.02);

    // test_mv_ell_vec_from_openfoam_coo_matrix("data/data.txt");

    // test_mv_ell_vec_from_matrix_market("data/mkt_real_symmetric/bcsstk04.mtx");

    // test_dot_product_vec();

    test_vec_axpy_vector();

    return 0;
}