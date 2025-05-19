#include <stdio.h>
#include <stdlib.h>

void print_dense_symmetric_matrix_from_csr(int n,
    double *diag,
    double *upper,
    int *col_index,
    int *row_ptr);

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

void mv_csr_symmetric(int n,                // dimensione matrice (n x n)
                      const double *diag,   // n valori sulla diagonale
                      const double *upper,  // valori non nulli triangolare superiore
                      const int *col_index, // colonne degli upper[]
                      const int *row_ptr,   // inizio riga in upper[] e col_index[]
                      const double *v,      // vettore di input
                      double *out);          // vettore di output

int generate_sparse_symmetric_csr(int n, double density,
                                      double **diag,
                                      double **upper,
                                      int **col_index,
                                      int **row_ptr);