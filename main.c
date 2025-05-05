#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include "parser.h"

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

// this function returns the value of the matrix at (i,j) position 
double get_matrix_entry(int i, int j, int n, double *diag, double *upper, int *i_indexes, int *j_indexes, int upper_count) {
    
    if (i == j) {   // diagonal case 
        return diag[i];  // return simply the i-th (or j-th) value of diag
    }

    // Matrix is symmetric, so A[i][j] == A[j][i]
    // switch i and j if i > j

    if (i > j){ // if we are in the lower triangle, switch i and j
        int temp = i;
        i = j;
        j = temp;
    }

    // Search in upper values
    for (int k = 0; k < upper_count; ++k) { //iterating all non zero upper values and corringponding indexes in i_indexes and j_indexes
        if (i_indexes[k] == i && j_indexes[k] == j) {   // if the indexes match with reqeusted (i,j)
            return upper[k];    // return the value
        }
    }

    // If not found the value is zero
    return 0.0;
}


int main(){

    const char *filename = "data.txt";
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error in opening file");
        return -1;
    }

    int count_diag = 0, count_upper = 0, count_lower = 0;

    // Parse all sections in a single file pass
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);

    // Print the first few values for each array
    if (diag) {
        printf("\ndiag( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf(")\n");
    }

    if (upper) {
        printf("\nupper( \n");
        for (int i = 0; i < 5 && i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_upper - 5; i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf(")\n");
    }

    if (upperAddr) {
        printf("\nupperAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf(")\n");
    }

    if(lowerAddr) {
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


    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);

    return 0;
}