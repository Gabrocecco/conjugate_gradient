#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "parser.h"
#include "vec.h"
#include "tests.h"

int conjugate_gradient(const int n, double *diag, int coo_length, double *upper,
                       int *rows, int *cols, double *b, double *x,
                       int max_iter, double tol);

/*
    Compile:
        gcc -std=c99 -Wall -pedantic -o main main.c tests.c parser.c vec.c -lm -g -fsanitize=address
    Run:
        ./main
    source          2000 ( 0.000000 0.000000 0.000000 0.000000 ...);
    diag            2000 ( -3.29086 -4.0237 -4.00375 -4.01 ...);
    upper           3890 ( 0.864978 1.21294 1.21294 0.943508, ...);
    upperAddr       3890 ( 1 10 190 2 11 191 3, ...);
    lowerAddr       3890 ( 0 0 0 1 1 1 2 2 2 3 3 3 4 4, ...);

*/

/* Solving Ax = b with CG method. */
int conjugate_gradient(const int n,    // matrix size (n x n)
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
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    // Initialize the solution vector x to zero
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }

    // Initialize the residual vector r_0 := b - A x_0
    mv_coo(n, coo_length, diag, upper, rows, cols, x, r); // A x_0
    vec_sub(b, r, r, n);                                  // r_0 = b - A x_0

    // Initialize the search direction p_0 := r_0
    vec_assign(p, r, n); // p_0 = r_0

    double r_dot_r_old = vec_dot(r, r, n); // r_k^T * r_k
    double r_dot_r_new = 0.0;

    for (int iter = 0; iter < max_iter; iter++)
    {

        mv_coo(n, coo_length, diag, upper, rows, cols, p, Ap); // Compute: A p_k
        double alpha = r_dot_r_old / vec_dot(p, Ap, n);        // alpha = (r^T * r) / (p^T * A * p)

        vec_axpy(x, p, alpha, x, n); // x_{k+1} = x_k + alpha * p_k

        vec_axpy(r, Ap, -alpha, r, n); // r_{k+1} = r_k - alpha * A * p_k

        r_dot_r_new = vec_dot(r, r, n); // r_{k+1}^T * r_{k+1}

        // Check for convergence
        if (sqrt(r_dot_r_new) < tol)
        {
            printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
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
        if (iter % 10 == 0)
            printf("Iteration %d: Residual norm = %.5e\n", iter, sqrt(r_dot_r_new));
    }

    printf("Maximum iterations reached without convergence.\n");
    printf("Residual norm: %.5e\n", sqrt(r_dot_r_new));
    // Free allocated memory
    free(r);
    free(p);
    free(Ap);

    return -1; // Return -1 to indicate failure to converge
}

int main()
{

    const char *filename = "data.txt";
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }

    int count_diag = 0, count_upper = 0, count_lower = 0;

    // Parse all sections in a single file pass
    double *source = parseDoubleArray(file, "source", &count_diag);
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);

    // Print the first few values for each array
    if (source)
    {
        printf("\n source( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf(")\n");
    }

    // Print the first few values for each array
    if (diag)
    {
        printf("\ndiag( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf(")\n");
    }

    if (upper)
    {
        printf("\nupper( \n");
        for (int i = 0; i < 5 && i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_upper - 5; i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf(")\n");
    }

    if (upperAddr)
    {
        printf("\nupperAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf(")\n");
    }

    if (lowerAddr)
    {
        printf("\nlowerAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf(")\n");
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------------- //
    int n_tests = 50;
    double time_spent_acc = 0;
    for(int i = 0; i < n_tests; i++)
    {   
        printf("Test number :%d\n", i);
        // initialize the output vector
        double *x = (double *)malloc(count_diag * sizeof(double));

        // Time the conjugate gradient method
        // Start the timer
        clock_t start = clock();
        int result = conjugate_gradient(
            count_diag,  // matrix size (n x n)
            diag,        // diagonal elements (exactly n dense elements)
            count_upper, // number of non-zero elements in the upper triangular part
            upper,       // non-zero elements in the upper triangular part
            upperAddr,   // i indexes of non-zero elements (upper)
            lowerAddr,   // j indexes of non-zero elements (upper)
            source,      // input vector
            x,           // output vector
            1000,        // maximum number of iterations
            1e-6);       // tolerance for convergence
        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        time_spent_acc += time_spent;
        printf("Time spent: %f seconds\n", time_spent);
        // if (result >= 0)
        // {
        //     // print some values of x
        //     for (int i = 0; i < 5 && i < count_diag; i++)
        //     {
        //         printf("x[%d] = %.10f\n", i, x[i]);
        //     }
        //     printf("...\n");
        //     // Print the last few values
        //     for (int i = count_diag - 5; i < count_diag; i++)
        //     {
        //         printf("x[%d] = %.10f\n", i, x[i]);
        //     }
        //     printf(")\n");
        // }
        // else
        // {
        //     printf("Didnt reach convergences init max_iter.\n");
        // }

        free(x);
    }
    printf("-------------------------------------------------------------------\n");
    printf("Average time spent in %d runs: %f seconds\n", n_tests, time_spent_acc / n_tests);

    // test_zero_matrix();

    // test_identity_matrix();

    // test_uniform_matrix();

    // test_sparse_symmetric_random_matrix();

    // Tests: -------------------------------------------------------------------------------------------------------------------------------------------- //

    // printing whole matrix
    // printf("Martix is: %d x %d\n", count_diag, count_diag);
    // for(int i; i < count_diag; i++){
    //     for(int j; j < count_diag; j++){
    //         printf("%lf ", get_matrix_entry(i, j, count_diag, diag, upper, upperAddr, lowerAddr, count_upper));
    //         if(i == j){
    //             printf("\n");
    //         }
    //     }
    // }

    // printf("Matrix diag: \n");
    // for(int i = 0; i < count_diag; i++){
    //     printf("%lf ", get_matrix_entry(i, i, count_diag, diag, upper, upperAddr, lowerAddr, count_upper));
    // }

    // test mv funcion with Ib = b
    // double *identity_diag = malloc(count_diag * sizeof(double));
    // // init identity_diag at all 1
    // for (int i = 0; i < count_diag; i++) {
    //     identity_diag[i] = 1.0;
    // }

    // double *input_vector = malloc(count_diag * sizeof(double));
    // // init input_vector from 0 to 1999
    // for (int i = 0; i < count_diag; i++) {
    //     input_vector[i] = (double)i;
    // }
    // print the input vector
    // printf("\nInput vector:\n");
    // for (int i = 0; i < count_diag; i++) {
    //     printf("input_vector[%d] = %f\n", i, input_vector[i]);
    // }

    // double *out = malloc(count_diag * sizeof(double));
    // mv_coo(count_diag, 0, identity_diag, NULL, NULL, NULL, input_vector, out);

    // print the result
    // printf("\nResult of mv_coo with identity matrix:\n");
    // for (int i = 0; i < count_diag; i++) {
    //     printf("out[%d] = %f\n", i, out[i]);
    // }

    // compare the result with the input vector
    // printf("\nComparing the result with the input vector:\n");
    // int error_flag=0;
    // for (int i = 0; i < count_diag; i++) {
    //     if (out[i] != input_vector[i]) {
    //         printf("out[%d] = %f, input_vector[%d] = %f\n", i, out[i], i, input_vector[i]);
    //         error_flag=1;
    //     }
    // }
    // if(error_flag==0){
    //     printf("All values are equal.\n");
    // }else{
    //     printf("Some values are not equal.\n");
    // }

    // count_upper = (count_diag * count_diag - count_diag) / 2;
    // printf("count_upper = %d\n", count_upper);

    // // test my function with the matrix with 1/n in every position
    // double *diag_test = malloc(count_diag * sizeof(double));
    // double *upper_test = malloc(count_upper * sizeof(double));
    // int *upperAddr_test = malloc(count_upper * sizeof(int));
    // int *lowerAddr_test = malloc(count_upper * sizeof(int));
    // double *input_vector_test = malloc(count_diag * sizeof(double));

    // // init diag with 1/n
    // double one_over_n = 1.0 / count_diag;
    // printf("count_diag = %d\n", count_diag);
    // printf("one_over_n = %f\n", one_over_n);

    // for (int i = 0; i < count_diag; i++)
    // {
    //     diag_test[i] = one_over_n;
    // }
    // // print the diag
    // printf("\nDiag:\n");
    // for (int i = 0; i < count_diag; i++)
    // {
    //     printf("diag_test[%d] = %f\n", i, diag_test[i]);
    // }
    // // print the sum of the diag
    // double sum_diag = 0.0;
    // for (int i = 0; i < count_diag; i++)
    // {
    //     sum_diag += diag_test[i];
    // }
    // printf("sum_diag = %lf\n", sum_diag);

    // int running_index = 0;
    // for (int i = 0; i < count_diag - 1; i++)
    // {
    //     for (int j = i + 1; j < count_diag; j++)
    //     {
    //         upper_test[running_index] = one_over_n;
    //         upperAddr_test[running_index] = i;
    //         lowerAddr_test[running_index] = j;
    //         running_index++;
    //     }
    // }

    // init input_vector with all 1
    // for (int i = 0; i < count_diag; i++)
    // {
    //     input_vector_test[i] = 1.0;
    // }
    // print the input vector
    // printf("\nInput vector:\n");
    // for (int i = 0; i < count_diag; i++) {
    //     printf("input_vector[%d] = %f\n", i, input_vector_test[i]);
    // }

    // double *out_test = malloc(count_diag * sizeof(double));
    // mv_coo(count_diag, count_upper, diag_test, upper_test, upperAddr_test, lowerAddr_test, input_vector_test, out_test);
    // // print the result
    // printf("\nResult of mv_coo with 1/n matrix:\n");
    // for (int i = 0; i < count_diag; i++)
    // {
    //     printf("out[%d] = %f\n", i, out_test[i]);
    // }

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(source);

    return 0;
}