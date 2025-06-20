#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <riscv_vector.h>

#ifdef __cplusplus
extern "C" {
#endif

void gen_rand_1d(double *a, int n);
void gen_string(char *s, int n);
void gen_rand_2d(double **ar, int n, int m);
void print_string(const char *a, const char *name);
void print_array_1d(double *a, int n, const char *type, const char *name);
void print_array_2d(double **a, int n, int m, const char *type, const char *name);
bool double_eq(double golden, double actual, double relErr);
bool compare_1d(double *golden, double *actual, int n);
bool compare_string(const char *golden, const char *actual, int n);
bool compare_2d(double **golden, double **actual, int n, int m);
double **alloc_array_2d(int n, int m);
void init_array_one_1d(double *ar, int n);
void init_array_one_2d(double **ar, int n, int m);
void print_dense_matrix_from_ell(int n, int max_nnz_row,
                                 const double *diag,
                                 const double *ell_values,
                                 const uint64_t *ell_cols);
void print_ell_format(int n, int max_nnz_row,
                      const double *ell_values,
                      const uint64_t *ell_cols);
int fp_eq(float reference, float actual, float relErr);
void print_mask_from_indices(const uint64_t *indices, size_t vl);
void print_vfloat64_vector(vfloat64m1_t vec, size_t vl, const char *label);
void print_vuint64_vector(vuint64m1_t vec, size_t vl, const char *label);
void print_vector(const double *vec, int n, const char *label);
void print_vuint64_vector_raw(vuint64m1_t vec, size_t vl, const char *label);

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
