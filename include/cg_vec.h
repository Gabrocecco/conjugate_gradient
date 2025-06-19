#include <stdio.h>
#include <string.h>
#include <stdint.h>

int conjugate_gradient_ell_full_colmajor_vectorized(
    const int   n,
    double     *diag,
    int         upper_count,
    double     *ell_values,
    uint64_t   *ell_col,
    int         max_nnz_row,
    double     *b,
    double     *x,
    int         max_iter,
    double      tol
);