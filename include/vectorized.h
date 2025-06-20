#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>

void mv_ell_symmetric_full_colmajor_vector(int n,
                                           int max_nnz_row, // max number of off-diagonal nnz in rows
                                           double *diag,
                                           double *ell_values, // ELL values (size n * max_nnz_row)
                                           uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                           double *x,          // input vector
                                           double *y);         // output vector

void mv_ell_symmetric_full_colmajor_vector_m2(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diangonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y);         // output vector

void mv_ell_symmetric_full_colmajor_vector_m4(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diagonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y);         // output vector

void mv_ell_symmetric_full_colmajor_vector_m8(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diagonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y);          // output vector
                                              
void mv_ell_symmetric_full_colmajor_vector_debug(int n,
                                                 int max_nnz_row, // max number of off-diagonal nnz in rows
                                                 double *diag,
                                                 double *ell_values, // ELL values (size n * max_nnz_row)
                                                 uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                                 double *x,          // input vector
                                                 double *y);         // output vector