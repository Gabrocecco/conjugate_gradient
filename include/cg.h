#include <stdio.h>
#include <stdlib.h>

int conjugate_gradient_coo(const int n, double *diag, int coo_length, double *upper,
                           int *rows, int *cols, double *b, double *x,
                           int max_iter, double tol);

int conjugate_gradient_csr(const int n,     // matrix size (n x n)
                           double *diag,    // diagonal elements (exactly n dense elements)
                           int upper_count, // number of non-zero elements in the upper triangular part
                           double *upper,   // non-zero elements in the upper triangular part
                           int *rows_ptr,   // starting index in upper[] for every row
                           int *cols,       // j indexes of non-zero elements in the upper triangular part
                           double *b,       // input vector
                           double *x,       // output vector
                           int max_iter,    // maximum number of iterations
                           double tol       // tolerance for convergence

);