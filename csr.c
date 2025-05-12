#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vec.h"

void generate_sparse_matrix(int m, int n, int nnz, int max_nnz_row, double *values, int *col_index, int *row_ptr) {
    srand(time(NULL));  // Initialize random seed
    int inserted = 0;   // Number of non-zero elements inserted
    row_ptr[0] = 0;     // Start of first row in CSR format

    for (int i = 0; i < m; i++) {
        // Random number of non-zero elements for this row
        int nnz_in_row = rand() % (max_nnz_row + 1);
        
        // Ensure that we do not insert more than nnz in total
        if (inserted + nnz_in_row > nnz) {
            nnz_in_row = nnz - inserted;
        }

        for (int j = 0; j < nnz_in_row; j++) {
            // Assign a random value between 1.0 and 10.0
            values[inserted] = (rand() % 10 + 1) + (rand() / (double)RAND_MAX); 
            
            // Random column index (could be optimized for no duplicates per row)
            col_index[inserted] = rand() % n;
            
            inserted++;
        }

        // Update row pointer to the next position in the CSR format
        row_ptr[i + 1] = inserted;

        // If all non-zero elements are inserted, mark the remaining rows as having no elements
        if (inserted >= nnz) {
            for (int k = i + 1; k < m; k++) {
                row_ptr[k + 1] = inserted;
            }
            break;
        }
    }
}

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

int mv_crs(double *values, int *col_index, int *row_ptr, double *x, double *y, int m, int n)
{   
    // for every row i 
    for(int i = 0; i < m; i++){ // i=1
        // take the first and last index of non-zero elements of that row in values[]
        int first = row_ptr[i]; 
        int last = row_ptr[i+1] - 1; 

        y[i] = 0.0; // inizializzo y[i] a 0
        // row i, is stored from row_ptr[i] to row_ptr[i+1]-1
        for(int j = first; j <= last; j++){ // we iterate from first to last index 
            y[i] += values[j] * x[col_index[j]];    // we multiply the value with the corresponding x value and we accumulate y[i]
        }
    }

    return 0;
}


int main(){
    // int m = 4;
    // int n = 5;
    // double values[] = {3.0, 22.0, 17.0, 7.0, 5.0, 8.0};
    // int col_index[] = {2, 0, 4, 0, 3, 1};
    // int row_ptr[] = {0, 1, 3, 5, 6};
    // double x[n];
    // double y[m];

    // // initialize y to 0
    // for(int i = 0; i < m; i++){
    //     y[i] = 0.0;
    // }
    // // initialize x to 0
    // for(int i = 0; i < n; i++){
    //     x[i] = (double)i;
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
    // printf("\n");

    // printf("Matrix A (4x5):\n");
    // for(int i = 0; i < m; i++){
    //     for(int j = 0; j < n; j++){
    //         printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
    //     }
    //     printf("\n");
    // }
    
    // // print values of x
    // printf("x: "); 
    // for(int i = 0; i < m; i++){
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

    int m = 10, n = 10, nnz = 100, max_nnz_row = 10;
    
    // Arrays to hold values, column indices, and row pointers
    double values[nnz];
    int col_index[nnz], row_ptr[m+1];

    // Generate the sparse matrix
    generate_sparse_matrix(m, n, nnz, max_nnz_row, values, col_index, row_ptr);

    // Print the CSR format representation of the matrix
    printf("Row pointers: ");
    for (int i = 0; i <= m; i++) {
        printf("%d ", row_ptr[i]);
    }
    printf("\n");

    printf("Column indices: ");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", col_index[i]);
    }
    printf("\n");

    printf("Values: ");
    for (int i = 0; i < nnz; i++) {
        printf("%f ", values[i]);
    }
    printf("\n");

    printf("Matrix A (CSR format):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
        }
        printf("\n");
    }

    return 0;

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

    return 0;
}
