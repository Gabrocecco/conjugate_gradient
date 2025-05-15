#include <stdio.h>
#include <stdlib.h>

int conjugate_gradient(const int n, double *diag, int coo_length, double *upper,
    int *rows, int *cols, double *b, double *x,
    int max_iter, double tol);