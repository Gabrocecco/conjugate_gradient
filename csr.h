#include <stdio.h>
#include <stdlib.h>

double get_matrix_entry_csr(int i, int j, int *row_ptr, int *col_index, double *values);

int mv_crs(double *values,  // where the non-zero elements are stored, row by row 
    int *col_index,  // column index of the non-zero elements
    int *row_ptr,    // row pointer, where the first element of each row is stored
    double *x,       // input vector (dimension = n)
    double *y,       // output vector (dimension = m)
    int m,           // number of rows of A 
    int n);           // number of columns of A


