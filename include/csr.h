#include <stdio.h>
#include <stdlib.h>

double get_matrix_entry_symmetric_csr(int i, int j, int n,
                                      double *diag,
                                      double *upper,
                                      int *col_index,
                                      int *row_ptr);

double get_generic_matrix_entry_csr(int i, int j, int nrows, int *row_ptr, int *col_index, double *values);

int mv_crs_generic(double *values, // where the non-zero elements are stored, row by row
                   int *col_index, // column index of the non-zero elements
                   int *row_ptr,   // row pointer, where the first element of each row is stored
                   double *x,      // input vector (dimension = n)
                   double *y,      // output vector (dimension = m)
                   int m,          // number of rows of A
                   int n);         // number of columns of A

int coo_to_csr(int triangular_dim,
               int upper_count,
               double *coo_upper_values,
               int *coo_row_indx,
               int *coo_col_indx,
               int *csr_row_ptr);

int generate_sparse_symmetric_csr(int n, double density,
                                  double **diag,
                                  double **upper,
                                  int **col_index,
                                  int **row_ptr);