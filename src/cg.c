#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "vec.h"
#include "cg.h"
#include "coo.h"
#include "csr.h"
#include "ell.h"

/*
    source          2000 ( 0.000000 0.000000 0.000000 0.000000 ...);
    diag            2000 ( -3.29086 -4.0237 -4.00375 -4.01 ...);
    upper           3890 ( 0.864978 1.21294 1.21294 0.943508, ...);
    upperAddr       3890 ( 1 10 190 2 11 191 3, ...);
    lowerAddr       3890 ( 0 0 0 1 1 1 2 2 2 3 3 3 4 4, ...);

*/

/* Solving Ax = b with CG method (COO) */
int conjugate_gradient_coo(const int n,    // matrix size (n x n)
                           double *diag,   // diagonal elements (exactly n dense elements)
                           int coo_length, // number of non-zero elements in the upper triangular part
                           double *upper,  // non-zero elements in the upper triangular part
                           int *rows,      // i indexes of non-zero elements in the upper triangular part
                           int *cols,      // j indexes of non-zero elements in the upper triangular part
                           double *b,      // input vector
                           double *x,      // output vector
                           int max_iter,   // maximum number of iterations
                           double tol      // tolerance for convergence

)
{
    printf("CG (COO) \n\n");
    double norm_factor = 2.0;
    double *r = (double *)malloc(n * sizeof(double));  // residaul vector r
    double *p = (double *)malloc(n * sizeof(double));  // direction vector p
    double *Ap = (double *)malloc(n * sizeof(double)); // projction vector A*p

    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    mv_coo_symmetric(n, coo_length, diag, upper, rows, cols, x, r); // A x_0
    vec_sub(b, r, r, n);                                            // r_0 = b - A x_0

    printf("r[0...4] = \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", r[i]);
    }
    printf("\n");

    // printf("Initial residual norm: %.5e\n", sqrt(vec_dot(r, r, n))); // Print the initial residual norm
    printf("Initial residual norm: %f\n", (vec_l1norm(r, n) / norm_factor)); // Print the initial residual norm

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 1; iter < max_iter; iter++)
    {
        mv_coo_symmetric(n, coo_length, diag, upper, rows, cols, p, Ap); // Compute: A p_k
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);                  // alpha = (r^T * r) / (p^T * A * p)

        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // If r_{k+1} is small enough, we stop
        // if (sqrt(r_dot_r_new) < tol)    // if sqrt(r_{k+1}^T * r_{k+1}) < tol
        if ((vec_l1norm(r, n) / norm_factor) < tol)
        {
            // printf("\nResidual norm: %.5e\n", sqrt(r_dot_r_new));
            printf("\nResidual norm: %.g\n", (vec_l1norm(r, n) / norm_factor));
            // Free allocated memory
            free(r);
            free(p);
            free(Ap);

            return iter + 1;
        }

        double beta = r_dot_r_new / r_dot_r_old; // beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

        vec_axpy(r, p, beta, p, n); // p_{k+1} = r_{k+1} + beta * p_k

        r_dot_r_old = r_dot_r_new; // Update r_dot_r_old for the next iteration

        // Print the residual norm every 10 iterations
        if (iter % 1 == 0 && iter > 0)
            // printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
            printf("Iteration %d: Residual norm = %g\n", iter, (vec_l1norm(r, n) / norm_factor));

        // printf("r[0...4] = \n");
        // for(int i=0; i < 5; i++){
        //     printf("%f ", r[i]);
        // }
        // printf("\n");
    }

    printf("Maximum iterations reached without convergence.\n");
    // printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    printf("Residual norm: %g\n", (vec_l1norm(r, n) / norm_factor));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}

/* Solving Ax = b with CG method (CSR) */
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

)
{
    printf("CG (CSR) \n\n");
    double norm_factor = 2.0;
    double *r = (double *)malloc(n * sizeof(double));  // residual vector r
    double *p = (double *)malloc(n * sizeof(double));  // direction vector p
    double *Ap = (double *)malloc(n * sizeof(double)); // projction vector A*p

    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    // mv_coo_symmetric(n, upper_count, diag, upper, rows_ptr, cols, x, r); // A x_0
    mv_csr_symmetric(n, diag, upper, cols, rows_ptr, x, r); // A x_0
    // printf("Debug: ")
    vec_sub(b, r, r, n); // r_0 = b - A x_0

    printf("r[0...4] = \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", r[i]);
    }
    printf("\n");

    // printf("Initial residual norm: %.5e\n", sqrt(vec_dot(r, r, n))); // Print the initial residual norm
    printf("Initial residual norm: %f\n", (vec_l1norm(r, n) / norm_factor)); // Print the initial residual norm

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 1; iter < max_iter; iter++)
    {
        // mv_coo_symmetric(n, upper_count, diag, upper, rows_ptr, cols, p, Ap); // Compute: A p_k
        mv_csr_symmetric(n, diag, upper, cols, rows_ptr, p, Ap); // Compute: A p_k
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);          // alpha = (r^T * r) / (p^T * A * p)

        // print alpha
        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // If r_{k+1} is small enough, we stop
        // if (sqrt(r_dot_r_new) < tol)    // if sqrt(r_{k+1}^T * r_{k+1}) < tol
        if ((vec_l1norm(r, n) / norm_factor) < tol)
        {
            // printf("\nResidual norm: %.5e\n", sqrt(r_dot_r_new));
            printf("\nResidual norm: %.g\n", (vec_l1norm(r, n) / norm_factor));
            // Free allocated memory
            free(r);
            free(p);
            free(Ap);

            return iter + 1;
        }

        double beta = r_dot_r_new / r_dot_r_old; // beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

        vec_axpy(r, p, beta, p, n); // p_{k+1} = r_{k+1} + beta * p_k

        r_dot_r_old = r_dot_r_new; // Update r_dot_r_old for the next iteration

        // Print the residual norm every 10 iterations
        if (iter % 1 == 0 && iter > 0)
            // printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
            printf("Iteration %d: Residual norm = %g\n", iter, (vec_l1norm(r, n) / norm_factor));

        // printf("r[0...4] = \n");
        // for(int i=0; i < 5; i++){
        //     printf("%f ", r[i]);
        // }
        // printf("\n");
    }

    printf("Maximum iterations reached without convergence.\n");
    // printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    printf("Residual norm: %g\n", (vec_l1norm(r, n) / norm_factor));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}

/* Solving Ax = b with CG method (ELL) */
int conjugate_gradient_ell(const int n,        // matrix size (n x n)
                           double *diag,       // diagonal elements (exactly n dense elements)
                           int upper_count,    // number of non-zero elements in the upper triangular part
                           double *ell_values, // non-zero elements in the upper triangular part
                           int *ell_col,       // starting index in upper[] for every row
                           int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                           double *b,          // input vector
                           double *x,          // output vector
                           int max_iter,       // maximum number of iterations
                           double tol          // tolerance for convergence
)
{
    printf("CG (ELL) \n\n");
    double norm_factor = 2.0;
    double *r = (double *)malloc(n * sizeof(double));  // residual vector r
    double *p = (double *)malloc(n * sizeof(double));  // direction vector p
    double *Ap = (double *)malloc(n * sizeof(double)); // projction vector A*p

    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, x, r); // A x_0
    vec_sub(b, r, r, n); // r_0 = b - A x_0

    printf("r[0...4] = \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", r[i]);
    }
    printf("\n");

    printf("Initial residual norm: %f\n", (vec_l1norm(r, n) / norm_factor)); // Print the initial residual norm

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 1; iter < max_iter; iter++)
    {
        mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, p, Ap); // Compute: A p_k
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);                           // alpha = (r^T * r) / (p^T * A * p)

        // print alpha
        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // If r_{k+1} is small enough, we stop
        // if (sqrt(r_dot_r_new) < tol)    // if sqrt(r_{k+1}^T * r_{k+1}) < tol
        if ((vec_l1norm(r, n) / norm_factor) < tol)
        {
            // printf("\nResidual norm: %.5e\n", sqrt(r_dot_r_new));
            printf("\nResidual norm: %.g\n", (vec_l1norm(r, n) / norm_factor));
            // Free allocated memory
            free(r);
            free(p);
            free(Ap);

            return iter + 1;
        }

        double beta = r_dot_r_new / r_dot_r_old; // beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

        vec_axpy(r, p, beta, p, n); // p_{k+1} = r_{k+1} + beta * p_k

        r_dot_r_old = r_dot_r_new; // Update r_dot_r_old for the next iteration

        // Print the residual norm every 10 iterations
        if (iter % 1 == 0 && iter > 0)
            // printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
            printf("Iteration %d: Residual norm = %g\n", iter, (vec_l1norm(r, n) / norm_factor));
    }

    printf("Maximum iterations reached without convergence.\n");
    // printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    printf("Residual norm: %g\n", (vec_l1norm(r, n) / norm_factor));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}


/* Solving Ax = b with CG method (ELL) */
int conjugate_gradient_ell_colmajor(const int n,        // matrix size (n x n)
                           double *diag,       // diagonal elements (exactly n dense elements)
                           int upper_count,    // number of non-zero elements in the upper triangular part
                           double *ell_values, // non-zero elements in the upper triangular part
                           int *ell_col,       // starting index in upper[] for every row
                           int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                           double *b,          // input vector
                           double *x,          // output vector
                           int max_iter,       // maximum number of iterations
                           double tol          // tolerance for convergence
)
{
    printf("CG (ELL) \n\n");
    double norm_factor = 2.0;
    double *r = (double *)malloc(n * sizeof(double));  // residual vector r
    double *p = (double *)malloc(n * sizeof(double));  // direction vector p
    double *Ap = (double *)malloc(n * sizeof(double)); // projction vector A*p

    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    // mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, x, r); // A x_0
    mv_ell_symmetric_upper_colmajor(n, max_nnz_row, diag, ell_values, ell_col, x, r); // A x_0
    vec_sub(b, r, r, n); // r_0 = b - A x_0

    printf("r[0...4] = \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", r[i]);
    }
    printf("\n");

    printf("Initial residual norm: %f\n", (vec_l1norm(r, n) / norm_factor)); // Print the initial residual norm

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 1; iter < max_iter; iter++)
    {
        // mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, p, Ap); // Compute: A p_k
        mv_ell_symmetric_upper_colmajor(n, max_nnz_row, diag, ell_values, ell_col, p, Ap); // Compute: A p_k
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);                           // alpha = (r^T * r) / (p^T * A * p)

        // print alpha
        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // If r_{k+1} is small enough, we stop
        // if (sqrt(r_dot_r_new) < tol)    // if sqrt(r_{k+1}^T * r_{k+1}) < tol
        if ((vec_l1norm(r, n) / norm_factor) < tol)
        {
            // printf("\nResidual norm: %.5e\n", sqrt(r_dot_r_new));
            printf("\nResidual norm: %.g\n", (vec_l1norm(r, n) / norm_factor));
            // Free allocated memory
            free(r);
            free(p);
            free(Ap);

            return iter + 1;
        }

        double beta = r_dot_r_new / r_dot_r_old; // beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

        vec_axpy(r, p, beta, p, n); // p_{k+1} = r_{k+1} + beta * p_k

        r_dot_r_old = r_dot_r_new; // Update r_dot_r_old for the next iteration

        // Print the residual norm every 10 iterations
        if (iter % 1 == 0 && iter > 0)
            // printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
            printf("Iteration %d: Residual norm = %g\n", iter, (vec_l1norm(r, n) / norm_factor));
    }

    printf("Maximum iterations reached without convergence.\n");
    // printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    printf("Residual norm: %g\n", (vec_l1norm(r, n) / norm_factor));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}

/* Solving Ax = b with CG method (ELL) */
int conjugate_gradient_ell_full_colmajor(const int n,        // matrix size (n x n)
                           double *diag,       // diagonal elements (exactly n dense elements)
                           int upper_count,    // number of non-zero elements in the upper triangular part
                           double *ell_values, // non-zero elements in the upper triangular part
                           int *ell_col,       // starting index in upper[] for every row
                           int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                           double *b,          // input vector
                           double *x,          // output vector
                           int max_iter,       // maximum number of iterations
                           double tol          // tolerance for convergence
)
{
    printf("CG (ELL) \n\n");
    double norm_factor = 2.0;
    double *r = (double *)malloc(n * sizeof(double));  // residual vector r
    double *p = (double *)malloc(n * sizeof(double));  // direction vector p
    double *Ap = (double *)malloc(n * sizeof(double)); // projction vector A*p

    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    // mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, x, r); // A x_0
    // mv_ell_symmetric_upper_colmajor(n, max_nnz_row, diag, ell_values, ell_col, x, r); // A x_0
    mv_ell_symmetric_full_colmajor(n, max_nnz_row, diag, ell_values, ell_col, x, r);
    vec_sub(b, r, r, n); // r_0 = b - A x_0

    printf("r[0...4] = \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", r[i]);
    }
    printf("\n");

    printf("Initial residual norm: %f\n", (vec_l1norm(r, n) / norm_factor)); // Print the initial residual norm

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 1; iter < max_iter; iter++)
    {
        // mv_ell_symmetric_upper(n, max_nnz_row, diag, ell_values, ell_col, p, Ap); // Compute: A p_k
        // mv_ell_symmetric_upper_colmajor(n, max_nnz_row, diag, ell_values, ell_col, p, Ap); // Compute: A p_k
        mv_ell_symmetric_full_colmajor(n, max_nnz_row, diag, ell_values, ell_col, p, Ap);
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);                           // alpha = (r^T * r) / (p^T * A * p)

        // print alpha
        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // If r_{k+1} is small enough, we stop
        // if (sqrt(r_dot_r_new) < tol)    // if sqrt(r_{k+1}^T * r_{k+1}) < tol
        if ((vec_l1norm(r, n) / norm_factor) < tol)
        {
            // printf("\nResidual norm: %.5e\n", sqrt(r_dot_r_new));
            printf("\nResidual norm: %.g\n", (vec_l1norm(r, n) / norm_factor));
            // Free allocated memory
            free(r);
            free(p);
            free(Ap);

            return iter + 1;
        }

        double beta = r_dot_r_new / r_dot_r_old; // beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

        vec_axpy(r, p, beta, p, n); // p_{k+1} = r_{k+1} + beta * p_k

        r_dot_r_old = r_dot_r_new; // Update r_dot_r_old for the next iteration

        // Print the residual norm every 10 iterations
        if (iter % 1 == 0 && iter > 0)
            // printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
            printf("Iteration %d: Residual norm = %g\n", iter, (vec_l1norm(r, n) / norm_factor));
    }

    printf("Maximum iterations reached without convergence.\n");
    // printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    printf("Residual norm: %g\n", (vec_l1norm(r, n) / norm_factor));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}