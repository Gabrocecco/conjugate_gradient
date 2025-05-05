#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parser.h"
// #include "linalgebra.h"

/*
    Compile:
        gcc -std=c99 -Wall -pedantic -o main main.c
    Run:
        ./main

    This is a parser for a loading a matrix data from file in coo format.
    Matrix is divided into:
        diag:
        upper:
        lower (symmetric)
    Indes are stored into:
        upperAddr: i index
        lowerAddr: j index

        matrix
{
    diag            2000 ( -3.29086 -4.0237 -4.00375 -4.01 ...);
    upper           3890 ( 0.864978 1.21294 1.21294 0.943508, ...);
    upperAddr       3890 ( 1 10 190 2 11 191 3, ...);
    lowerAddr       3890 ( 0 0 0 1 1 1 2 2 2 3 3 3 4 4, ...);

*/

// COO-dense openfoam hybrid
/*

*/
void mv_coo(int n,               // matrix size (n x n)
            int coo_length,      // number of non-zero elements in the upper triangular part
            const double *diag,  // diagonal elements (exactly n dense elements)
            const double *value, // non-zero elements in the upper triangular part
            const int *rows,     // i indexes of non-zero elements (value)
            const int *columns,  // j indexes of non-zero elements (value)
            const double *v,     // input vector
            double *out)
{ // output vector

    // init output vector
    for (int i = 0; i < n; i++)
    {
        out[i] = 0.0;
    }

    // dense diagonal
    for (int i = 0; i < n; i++)
    {
        out[i] += diag[i] * v[i]; // accumulate the contribution of the diagonal
    }

    // upper triangular matrix
    for (int element_index = 0; element_index < coo_length; element_index++)
    {                                                    // iterate over all non-zero elements in values (top triangular part, by row)
        const int row_index = rows[element_index];       // i, actual row index of A
        const int column_index = columns[element_index]; // j, actual column index of A
        const double val = value[element_index];         // A[i][j]
        out[row_index] += val * v[column_index];         // out[i] += A[i][j] * v[j]
        out[column_index] += val * v[row_index];         // out[j] += A[j][i] * v[i]   // symmetric contribution
    }
}

// this function returns the value of the matrix at (i,j) position
double get_matrix_entry(int i, int j, int n, double *diag, double *upper, int *i_indexes, int *j_indexes, int upper_count)
{

    if (i == j)
    {                   // diagonal case
        return diag[i]; // return simply the i-th (or j-th) value of diag
    }

    // Matrix is symmetric, so A[i][j] == A[j][i]
    // switch i and j if i > j

    if (i > j)
    { // if we are in the lower triangle, switch i and j
        int temp = i;
        i = j;
        j = temp;
    }

    // Search in upper values
    for (int k = 0; k < upper_count; ++k)
    { // iterating all non zero upper values and corringponding indexes in i_indexes and j_indexes
        if (i_indexes[k] == i && j_indexes[k] == j)
        {                    // if the indexes match with reqeusted (i,j)
            return upper[k]; // return the value
        }
    }

    // If not found the value is zero
    return 0.0;
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
        for (int i = 0; i < count_diag; i++)
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

    count_upper = (count_diag * count_diag - count_diag) / 2;
    printf("count_upper = %d\n", count_upper);

    // test my function with the matrix with 1/n in every position
    double *diag_test = malloc(count_diag * sizeof(double));
    double *upper_test = malloc(count_upper * sizeof(double));
    int *upperAddr_test = malloc(count_upper * sizeof(int));
    int *lowerAddr_test = malloc(count_upper * sizeof(int));
    double *input_vector_test = malloc(count_diag * sizeof(double));

    // init diag with 1/n
    double one_over_n = 1.0 / count_diag;
    printf("count_diag = %d\n", count_diag);
    printf("one_over_n = %f\n", one_over_n);

    for (int i = 0; i < count_diag; i++)
    {
        diag_test[i] = one_over_n;
    }
    // print the diag
    printf("\nDiag:\n");
    for (int i = 0; i < count_diag; i++)
    {
        printf("diag_test[%d] = %f\n", i, diag_test[i]);
    }
    // print the sum of the diag
    double sum_diag = 0.0;
    for (int i = 0; i < count_diag; i++)
    {
        sum_diag += diag_test[i];
    }
    printf("sum_diag = %lf\n", sum_diag);

    int running_index = 0;
    for (int i = 0; i < count_diag - 1; i++)
    {
        for (int j = i + 1; j < count_diag; j++)
        {
            upper_test[running_index] = one_over_n;
            upperAddr_test[running_index] = i;
            lowerAddr_test[running_index] = j;
            running_index++;
        }
    }

   /* // init upper with 1/n
    for (int i = 0; i < count_upper; i++)
    {
        upper_test[i] = 1.0 / one_over_n;
    }
    // init upperAddr with 1/n
    for (int i = 0; i < count_upper - 1; i++)
    {
        upperAddr_test[i] = i + 1;
    }
    // init lowerAddr with 1/n
    for (int i = 0; i < count_upper - 1; i++)
    {
        lowerAddr_test[i] = i + 1;
    }*/
    // init input_vector with all 1
    for (int i = 0; i < count_diag; i++)
    {
        input_vector_test[i] = 1.0;
    }
    // print the input vector
    // printf("\nInput vector:\n");
    // for (int i = 0; i < count_diag; i++) {
    //     printf("input_vector[%d] = %f\n", i, input_vector_test[i]);
    // }

    double *out_test = malloc(count_diag * sizeof(double));
    mv_coo(count_diag, count_upper, diag_test, upper_test, upperAddr_test, lowerAddr_test, input_vector_test, out_test);
    // print the result
    printf("\nResult of mv_coo with 1/n matrix:\n");
    for (int i = 0; i < count_diag; i++)
    {
        printf("out[%d] = %f\n", i, out_test[i]);
    }
    

    // void mv_coo(int n,               // matrix size (n x n)
    //         int coo_length,      // number of non-zero elements in the upper triangular part
    //         const double* diag,  // diagonal elements (exactly n dense elements)
    //         const double* value, // non-zero elements in the upper triangular part
    //         const int* rows,     // i indexes of non-zero elements (value)
    //         const int* columns,  // j indexes of non-zero elements (value)
    //         const double* v,     // input vector
    //         double* out) {       // output vector

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);

    return 0;
}