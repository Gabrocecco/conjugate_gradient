#ifndef ELL_H
#define ELL_H

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

void analyze_ell_matrix(int n, int nnz_max, double *ell_values, int *ell_col_idx);

void coo_to_ell_symmetric_upper_colmajor(int n,                           // dimension of matrix A (n x n)
                                         int upper_nnz,                   // number of non-zeros in upper triangular part
                                         double coo_values_upper[], // values of upper triangular part
                                         int coo_rows[],            // row indices of upper triangular part
                                         int coo_cols[],            // column indices of upper triangular part
                                         double ell_values[],             // ELL values array (to be filled; size = n * max_nnz_row)
                                         int ell_cols[],                  // ELL column indices array (to be filled; same size)
                                         int max_nnz_row                  // maximum number of non-zeros per row in ELL format
);

void mv_ell_symmetric_upper_colmajor(int n,                     // dimension of matrix A (n x n)
                                     int max_nnz_row,           // maximum number of non-zeros per row in ELL format
                                     double diag[],       // diagonal elements (size n)
                                     double ell_values[], // ELL values array (size n * max_nnz_row)
                                     int ell_cols[],      // ELL column indices array (size n * max_nnz_row)
                                     double x[],          // input vector (size n)
                                     double y[]                 // output vector (size n)
);

void analyze_ell_matrix_colmajor(int n, int nnz_max,
                                 const double ell_values[],
                                 const int    ell_col_idx[]);
#endif // ELLPACK_H
