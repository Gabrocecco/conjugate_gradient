#include <stdio.h>
#include <stdlib.h>

void generate_sparse_symmetric_coo(int n, double sparsity,
                                  int *upper_nnz,
                                  double **coo_values_upper,
                                  int **coo_rows,
                                  int **coo_cols,
                                  double **diagonal);

// this function returns the value of the matrix at (i,j) position
double get_matrix_entry_symmetric_coo(int i,
                                      int j,
                                      int n,
                                      double *diag,
                                      double *upper,
                                      int *i_indexes,
                                      int *j_indexes,
                                      int upper_count);

int generate_sparse_symmetric_matrix_full_diag_coo(int n, double density, double **diag, double **upper, int **upperAddr, int **lowerAddr);

void mv_coo_symmetric(int n,               // matrix size (n x n)
            int coo_length,      // number of non-zero elements in the upper triangular part
            const double *diag,  // diagonal elements (exactly n dense elements)
            const double *value, // non-zero elements in the upper triangular part
            const int *rows,     // i indexes of non-zero elements (value)
            const int *columns,  // j indexes of non-zero elements (value)
            const double *v,     // input vector
            double *out);

void print_dense_symmeric_matrix_from_coo(int n, double *diag, double *upper, int *row_inx, int *col_inx, int upper_count);

int compare_symmetric_matrices_coo_csr(int n,
    double *diag_coo,
    double *upper_coo,
    int *i_idx,
    int *j_idx,
    int upper_count,
    double *diag_csr,
    double *upper_csr,
    int *col_index,
    int *row_ptr);