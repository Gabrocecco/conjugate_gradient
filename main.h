#ifndef MAIN_H
#define MAIN_H

int conjugate_gradient(const int n, double *diag, int coo_length, double *upper,
                       int *rows, int *cols, double *b, double *x,
                       int max_iter, double tol);

#endif // MAIN_H