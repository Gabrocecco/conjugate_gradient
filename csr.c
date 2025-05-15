#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vec.h"

// define a CSR matrix 
/*        A              x
    0  0  3  0  0        1
    22 0  0  0  17       2
    7  0  0  5  0        3 
    0  8  0  0  0        4
                         5
values = [3, 22, 17, 7, 5, 8]
col_index = [2, 0, 4, 0, 3, 1]
row_ptr = [0, 1, 3, 5, 6]
*/

double get_matrix_entry_csr(int i, int j, int *row_ptr, int *col_index, double *values) {
    // Check if the row index is valid
    if (i < 0) {
        printf("Invalid row index\n");
        return -1000; // Invalid row index
    }

    // Get the start and end indices for the row
    int first = row_ptr[i];
    int last = row_ptr[i + 1] - 1;

    // Search for the column index in the specified row
    for (int k = first; k <= last; k++) {
        if (col_index[k] == j) {
            return values[k]; // Return the value if found
        }
    }

    return 0.0; // Return 0 if not found
}

// matrix vector multiplication of a m x n matrix in CSR format
/* 
   for i = 1, n 
       y(i)  = 0 
       for j = row_ptr(i), row_ptr(i+1) - 1
           y(i) = y(i) + val(j) * x(col_ind(j))
       end;
   end;
*/
int mv_crs(double *values,  // where the non-zero elements are stored, row by row 
           int *col_index,  // column index of the non-zero elements
           int *row_ptr,    // row pointer, where the first element of each row is stored
           double *x,       // input vector (dimension = n)
           double *y,       // output vector (dimension = m)
           int m,           // number of rows of A 
           int n)           // number of columns of A
{   
    // for every row i 
    for(int i = 0; i < m; i++){ 
        printf("row %d:\n", i);
        // take the first and last index of non-zero elements of that row in values[]
        int first = row_ptr[i]; // location index in values[] of the first non-zero element of row i
        int last = row_ptr[i+1] - 1; // location index in values[] of the last non-zero element of row i
        // print the first and last index
        printf("Elements of row %d are stored in values[] in positions:\n ", i);
        printf("first: %d, last: %d\n", first, last);
        y[i] = 0.0;
        if(first > last)
            printf("row %d has 0 non-zero elements, we fix y[%d] = 0.0 \n", i, i);
        for(int j = first; j <= last; j++){ // we iterate from first to last index 
            y[i] += values[j] * x[col_index[j]];    // we multiply the value with the corresponding x value and we accumulate y[i]
            printf("y[%d] += %f * x[%d] = %f * %f = %f\n", i, values[j], col_index[j], values[j], x[col_index[j]], y[i]);
        }
    }

    return 0;
}


// int main(){
    // define a CSR matrix 
    /*        A              x
        0  0  3  0  0        1
        22 0  0  0  17       2
        7  0  0  5  0        3 
        0  8  0  0  0        4
                            5
    values = [3, 22, 17, 7, 5, 8]
    col_index = [2, 0, 4, 0, 3, 1]
    row_ptr = [0, 1, 3, 5, 6]
    */

    // int m = 4;
    // int n = 5;
    // double values[] = {3.0, 22.0, 17.0, 7.0, 5.0, 8.0};
    // int col_index[] = {2, 0, 4, 0, 3, 1};
    // int row_ptr[] = {0, 1, 3, 5, 6};
    // double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    // double y[m];

    // define a CSR matrix 
    /*        A              x
        0  0  3  0  0        1
        22 0  0  0  17       2
        0  0  0  0  0        3 
        0  8  0  0  0        4
                             5
    values =    [3, 22, 17, 8]
    col_index = [2, 0,  4,  1]
    row_ptr =   [0, 1,  3,  3, 4]
    */

    // int m = 4;
    // int n = 5;
    // double values[] = {3, 22, 17, 8};
    // int col_index[] = {2, 0,  4,  1};
    // int row_ptr[] = {0, 1,  3,  3, 4};
    // double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    // double y[m];

    // // initialize y to 0
    // for(int i = 0; i < m; i++){
    //     y[i] = 0.0;
    // }

    // //print values[]
    // printf("values: ");
    // for(int i = 0; i < 6; i++){
    //     printf("%f ", values[i]);
    // }
    // printf("\n");
    // // print col_index[]
    // printf("col_index: ");
    // for(int i = 0; i < 6; i++){
    //     printf("%d ", col_index[i]);
    // }
    // printf("\n");
    // // print row_ptr[]
    // printf("row_ptr: ");
    // for(int i = 0; i < 5; i++){
    //     printf("%d ", row_ptr[i]);
    // }
    // printf("\n\n");

    // printf("Matrix A (4x5):\n");
    // for(int i = 0; i < m; i++){
    //     for(int j = 0; j < n; j++){
    //         printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
    //     }
    //     printf("\n");
    // }
    
    // // print values of x
    // printf("\nx: "); 
    // for(int i = 0; i < n; i++){
    //     printf("%f ", x[i]);
    // }
    // printf("\n");


    // // call the function
    // mv_crs(values, col_index, row_ptr, x, y, m, n);
    // // print the result
    // printf("y: ");
    // for(int i = 0; i < m; i++){
    //     printf("%f ", y[i]);
    // }
    // printf("\n");

    // return 0;

    /*
    Matrix A (6x6):
    0 0 0 0 0 9
    5 0 0 0 0 0
    0 0 2 0 0 0
    0 3 0 0 0 0
    0 0 0 0 7 0
    0 0 0 4 0 0
    */

    // double values[] = {9.0, 5.0, 2.0, 3.0, 7.0, 4.0};
    // int col_index[] = {5, 0, 2, 1, 4, 3};
    // int row_ptr[] = {0, 1, 2, 3, 4, 5, 6};  // lunghezza = n_righe + 1 = 7
    // double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // double y[6];

    // mv_crs(values, col_index, row_ptr, x, y, 6, 6);
    // // print the result
    // printf("y: ");
    // for(int i = 0; i < 6; i++){
    //     printf("%f ", y[i]);
    // }
    // printf("\n");

    // printf("Matrix A (6x6):\n");
    // for(int i = 0; i < 6; i++){
    //     for(int j = 0; j < 6; j++){
    //         printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
    //     }
    //     printf("\n");
    // }

    // int m = 10, n = 10, nnz = 100, max_nnz_row = 10;
    
    // // Arrays to hold values, column indices, and row pointers
    // double values[nnz];
    // int col_index[nnz], row_ptr[m+1];

    // // Generate the sparse matrix
    // generate_sparse_matrix(m, n, nnz, max_nnz_row, values, col_index, row_ptr);

    // // Print the CSR format representation of the matrix
    // printf("Row pointers: ");
    // for (int i = 0; i <= m; i++) {
    //     printf("%d ", row_ptr[i]);
    // }
    // printf("\n");

    // printf("Column indices: ");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%d ", col_index[i]);
    // }
    // printf("\n");

    // printf("Values: ");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%f ", values[i]);
    // }
    // printf("\n");

    // printf("Matrix A (CSR format):\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
    //     }
    //     printf("\n");
    // }

    // return 0;

    // // Vettore di input x
    // double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // // Vettore di output y
    // double y[m];

    // // Calcola il prodotto matrice-vettore
    // mv_crs(values, col_index, row_ptr, x, y, m, n);

    // // Stampa il risultato
    // printf("Risultato del prodotto matrice-vettore:\n");
    // for (int i = 0; i < m; i++) {
    //     printf("y[%d] = %.2f\n", i, y[i]);
    // }

    

//     return 0;
// }
