#ifndef ELL_H
#define ELL_H

#include <stdint.h> 
#include <stddef.h>

void print_dense_matrix(double **A, int n);

void convert_ell_to_dense(int n, double *diag, double *ell_values, int *ell_cols, int max_nnz_row, double **A);

void convert_ell_to_dense_colmajor(int n, double *diag, double *ell_values, int *ell_cols, int max_nnz_row, double **A);

void convert_ell_full_to_dense_rowmajor(int n,
                                        const double diag[],
                                        const double ell_values[],
                                        const int ell_cols[],
                                        int max_nnz_row,
                                        double **A);

void convert_ell_full_to_dense_colmajor(int n,
                                        const double diag[],
                                        const double ell_values[],
                                        const int ell_cols[],
                                        int max_nnz_row,
                                        double **A);

void coo_to_ell_symmetric_full_colmajor_sdtint(int n,
                                               int upper_nnz,
                                               double *coo_values_upper,
                                               int *coo_rows,
                                               int *coo_cols,
                                               double *ell_values,
                                               uint64_t *ell_cols,
                                               int max_nnz_row);

// Generic CSR to ELLPACK conversion
void csr_to_ell_generic(int n, int nnz,
                        double *csr_values,
                        int *csr_row_ptr,
                        int *csr_cols,
                        double *ell_values,
                        int *ell_cols);

// CSR (symmetric upper + separate diagonal) to ELLPACK (upper only + diagonal first)
void csr_to_ell_symmetric_only_upper(int n,
                                     double *diag,
                                     int upper_nnz,
                                     double *csr_values,
                                     int *csr_row_ptr,
                                     int *csr_cols,
                                     double *ell_values,
                                     int *ell_cols);

// CSR (symmetric upper + separate diagonal) to full ELLPACK format (symmetric expansion)
void csr_to_ell_symmetric_full(int n,
                               double *diag,
                               int upper_nnz,
                               double *csr_values,
                               int *csr_row_ptr,
                               int *csr_cols,
                               double *ell_values,
                               int *ell_cols);

// COO (symmetric upper + separate diagonal) to full ELLPACK format (symmetric expansion)
void coo_to_ell_symmetric_full(int n,
                               double *diag,
                               int upper_nnz,
                               double *coo_values,
                               int *coo_rows,
                               int *coo_cols,
                               double *ell_values,
                               int *ell_cols);

// COO (symmetric upper + separate diagonal) to ELLPACK storing only upper part (diagonal excluded)
void coo_to_ell_symmetric_upper(int n,
                                int upper_nnz,
                                double *coo_values,
                                int *coo_rows,
                                int *coo_cols,
                                double *ell_values,
                                int *ell_cols,
                                int max_nnz_row);

// Matrix-vector multiplication for generic ELLPACK matrix
void mv_ell(int n, int max_nnz_row,
            double *ell_values, int *ell_cols,
            double *x, double *y);

void mv_ell_symmetric_upper_opt(int n,              // dimension of matrix A (n x n)
                                int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                                double *diag,       // diagonal elements (size n)
                                double *ell_values, // upper triangular values only, with padding (size n * max_nnz_row)
                                int *ell_cols,      // column indices (same layout as ell_values)
                                double *x,          // input vector (size n)
                                double *y           // output vector (size n)
);

// Matrix-vector multiplication for symmetric matrix in ELLPACK format (upper part + diagonal separate)
void mv_ell_symmetric_upper(int n, int max_nnz_row,
                            double *diag,
                            double *ell_values,
                            int *ell_cols,
                            double *x,
                            double *y);

int compute_max_nnz_row_upper(int n, int upper_nnz, int *coo_rows, int *coo_cols);

int compute_max_nnz_row_full(int n,
                             int upper_nnz,
                             const int *coo_rows,
                             const int *coo_cols);

void analyze_ell_matrix(int n, int nnz_max, double *ell_values, int *ell_col_idx);

void coo_to_ell_symmetric_upper_colmajor(int n,                    // dimension of matrix A (n x n)
                                         int upper_nnz,            // number of non-zeros in upper triangular part
                                         double *coo_values_upper, // values of upper triangular part
                                         int *coo_rows,            // row indices of upper triangular part
                                         int *coo_cols,            // column indices of upper triangular part
                                         double *ell_values,       // ELL values array (to be filled; size = n * max_nnz_row)
                                         int *ell_cols,            // ELL column indices array (to be filled; same size)
                                         int max_nnz_row           // maximum number of non-zeros per row in ELL format
);

void coo_to_ell_symmetric_full_colmajor(int n,
                                        int upper_nnz,
                                        double *coo_values_upper,
                                        int *coo_rows,
                                        int *coo_cols,
                                        double *ell_values,
                                        int *ell_cols,
                                        int max_nnz_row);

void mv_ell_symmetric_upper_colmajor(int n,              // dimension of matrix A (n x n)
                                     int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                                     double *diag,       // diagonal elements (size n)
                                     double *ell_values, // ELL values array (size n * max_nnz_row)
                                     int *ell_cols,      // ELL column indices array (size n * max_nnz_row)
                                     double *x,          // input vector (size n)
                                     double *y           // output vector (size n)
);

void mv_ell_symmetric_full_colmajor(int n,              // dimension of matrix A (n x n)
                                    int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                                    double *diag,       // diagonal elements (size n)
                                    double *ell_values, // ELL values array (size n * max_nnz_row)
                                    int *ell_cols,      // ELL column indices array (size n * max_nnz_row)
                                    double *x,          // input vector (size n)
                                    double *y           // output vector (size n)
);

void mv_ell_symmetric_full_colmajor_sdtint(int n,
                                           int max_nnz_row, // max number of off-diagonal nnz in rows
                                           double *diag,
                                           double *ell_values, // ELL values (size n * max_nnz_row)
                                           uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                           double *x,          // input vector
                                           double *y);         // output vector

void analyze_ell_matrix_colmajor(int n, int nnz_max,
                                 const double *ell_values,
                                 const int *ell_col_idx);

double analyze_ell_matrix_full_colmajor(int n, int nnz_max,
                                      const double *ell_values,
                                      const int *ell_col_idx);

#endif // ELLPACK_H
