// common.h
// common utilities for the test code under exmaples/

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void gen_rand_1d(double *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
}

void gen_string(char *s, int n) {
  // char value range: -128 ~ 127
  for (int i = 0; i < n - 1; ++i)
    s[i] = (char)(rand() % 127) + 1;
  s[n - 1] = '\0';
}

void gen_rand_2d(double **ar, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      ar[i][j] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
}

void print_string(const char *a, const char *name) {
  printf("const char *%s = \"", name);
  int i = 0;
  while (a[i] != 0)
    putchar(a[i++]);
  printf("\"\n");
  puts("");
}

void print_array_1d(double *a, int n, const char *type, const char *name) {
  printf("%s %s[%d] = {\n", type, name, n);
  for (int i = 0; i < n; ++i) {
    printf("%06.2f%s", a[i], i != n - 1 ? "," : "};\n");
    if (i % 10 == 9)
      puts("");
  }
  puts("");
}

void print_array_2d(double **a, int n, int m, const char *type,
                    const char *name) {
  printf("%s %s[%d][%d] = {\n", type, name, n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      printf("%06.2f", a[i][j]);
      if (j == m - 1)
        puts(i == n - 1 ? "};" : ",");
      else
        putchar(',');
    }
  }
  puts("");
}

bool double_eq(double golden, double actual, double relErr) {
  return (fabs(actual - golden) < relErr);
}

bool compare_1d(double *golden, double *actual, int n) {
  for (int i = 0; i < n; ++i)
    if (!double_eq(golden[i], actual[i], 1e-6))
      return false;
  return true;
}

bool compare_string(const char *golden, const char *actual, int n) {
  for (int i = 0; i < n; ++i)
    if (golden[i] != actual[i])
      return false;
  return true;
}

bool compare_2d(double **golden, double **actual, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if (!double_eq(golden[i][j], actual[i][j], 1e-6))
        return false;
  return true;
}

double **alloc_array_2d(int n, int m) {
  double **ret;
  ret = (double **)malloc(sizeof(double *) * n);
  for (int i = 0; i < n; ++i)
    ret[i] = (double *)malloc(sizeof(double) * m);
  return ret;
}

void init_array_one_1d(double *ar, int n) {
  for (int i = 0; i < n; ++i)
    ar[i] = 1;
}

void init_array_one_2d(double **ar, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      ar[i][j] = 1;
}

void print_dense_matrix_from_ell(int n, int max_nnz_row,
                                 const double *diag,
                                 const double *ell_values,
                                 const uint64_t *ell_cols)
{
    // Allocate dense matrix A[n][n] as flat array
    double *A = calloc(n * n, sizeof(double));
    if (!A) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return;
    }

    // Fill diagonal
    for (int i = 0; i < n; ++i)
        A[i * n + i] = diag[i];

    // Fill off-diagonal
    for (int j = 0; j < max_nnz_row; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            int idx = j * n + i;
            int col = ell_cols[idx];
            double v = ell_values[idx];

            if (col == -1)
                continue;

            A[i * n + col] += v;
        }
    }

    // Print matrix
    printf("Dense matrix A (reconstructed from FULL-ELL):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%6.1f ", A[i * n + j]);
        printf("\n");
    }
    printf("\n");

    free(A);
}

void print_ell_format(int n, int max_nnz_row,
                      const double *ell_values,
                      const uint64_t *ell_cols)
{
    printf("ELL-FULL values and column indices (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        printf("Row %d: values = [", i);
        for (int j = 0; j < max_nnz_row; ++j)
        {
            int idx = j * n + i;
            printf("%4.1f%s", ell_values[idx], (j < max_nnz_row - 1) ? ", " : "");
        }
        printf("], cols = [");
        for (int j = 0; j < max_nnz_row; ++j)
        {
            int idx = j * n + i;
            if (ell_cols[idx] == (uint64_t)-1)
                printf("  -1%s", (j < max_nnz_row - 1) ? ", " : "");
            else
                printf("%4llu%s", (unsigned long long)ell_cols[idx], (j < max_nnz_row - 1) ? ", " : "");
        }
        printf("]\n");
    }
    printf("\n");
}

int fp_eq(float reference, float actual, float relErr)
{
    // if near zero, do absolute error instead.
    float absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
    return fabsf(actual - reference) < absErr;
}

void print_mask_from_indices(const uint64_t *indices, size_t vl)
{
    printf("mask[] (1 = valid, 0 = padding):\n");
    for (size_t i = 0; i < vl; ++i)
    {
        uint64_t col = indices[i];
        printf("%d\n", col != (uint64_t)-1 ? 1 : 0);
    }
    printf("\n");
}

void print_vfloat64_vector(vfloat64m1_t vec, size_t vl, const char *label)
{
    double buffer[vl];  // buffer temporaneo
    __riscv_vse64_v_f64m1(buffer, vec, vl);  // store del vettore in memoria

    printf("%s:\n", label);
    for (size_t i = 0; i < vl; ++i)
        printf("%.2f\n", buffer[i]);
    printf("\n");
}

void print_vuint64_vector(vuint64m1_t vec, size_t vl, const char *label)
{
    uint64_t buffer[vl];  // buffer temporaneo
    __riscv_vse64_v_u64m1(buffer, vec, vl);  // store del vettore in memoria

    printf("%s:\n", label);
    for (size_t i = 0; i < vl; ++i)
        printf("%4ld\n", (long)buffer[i]);  // usa %ld per evitare warning
    printf("\n");
}

void print_vector(const double *vec, int n, const char *label)
{
    printf("%s:\n", label);
    for (int i = 0; i < n; ++i)
    {
        printf("  x[%d] = %g\n", i, vec[i]);
    }
    printf("\n");
}

void print_vuint64_vector_raw(vuint64m1_t vec, size_t vl, const char *label)
{
    uint64_t temp[vl];
    __riscv_vse64_v_u64m1(temp, vec, vl);
    printf("%s (raw hex):\n", label);
    for (size_t i = 0; i < vl; i++)
        printf("  [%zu] = 0x%016lx\n", i, temp[i]);
}