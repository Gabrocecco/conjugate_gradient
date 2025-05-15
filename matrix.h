#include <stdio.h>
#include <stdlib.h>

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
