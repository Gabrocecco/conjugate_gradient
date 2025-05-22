#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vec.h"
#include "utils.h"

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

double get_generic_matrix_entry_csr(int i, int j, int nrows, int *row_ptr, int *col_index, double *values)
{
    if (i < 0 || i >= nrows || j < 0)
    {
        fprintf(stderr, "Invalid index (%d, %d)\n", i, j);
        return NAN; // Not a Number
    }

    // Get the start and end indices for the row
    int first = row_ptr[i];
    int last = row_ptr[i + 1] - 1;

    // Search for the column index in the specified row
    for (int k = first; k <= last; k++)
    {
        if (col_index[k] == j)
        {
            return values[k]; // Return the value if found
        }
    }

    return 0.0; // Return 0 if not found
}

double get_matrix_entry_symmetric_csr(int i, int j, int n,
                                      double *diag,
                                      double *upper,
                                      int *col_index,
                                      int *row_ptr)
{
    if (i < 0 || j < 0 || i >= n || j >= n)
    {
        fprintf(stderr, "Invalid index (%d, %d)\n", i, j);
        return NAN;
    }

    if (i == j)
    {
        return diag[i];
    }

    if (i > j)
    { // if we are in the lower triangle, switch i and j
        int temp = i;
        i = j;
        j = temp;
    }

    int start = row_ptr[i]; // we want to iterate only in the portion of upper[] where (i,j) is located, we use row_ptr[] to do that
    int end = row_ptr[i + 1];

    for (int k = start; k < end; k++)
    {
        if (col_index[k] == j)
        {
            return upper[k];
        }
    }

    return 0.0; // implicit value zero
}

void print_dense_symmetric_matrix_from_csr(int n,
                                           double *diag,
                                           double *upper,
                                           int *col_index,
                                           int *row_ptr)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double value = get_matrix_entry_symmetric_csr(i, j, n, diag, upper, col_index, row_ptr);
            printf("%f ", value);
        }
        printf("\n");
    }
}

// Generate a sparse symmetric matrix with dense diag with a parameter density
int generate_sparse_symmetric_csr(int n, double density,
                                  double **diag,
                                  double **upper,
                                  int **col_index,
                                  int **row_ptr)
{
    // Memory allocaion
    *diag = (double *)malloc(n * sizeof(double));
    *upper = (double *)malloc(n * n * sizeof(double)); // spazio massimo
    *col_index = (int *)malloc(n * n * sizeof(int));
    *row_ptr = (int *)malloc((n + 1) * sizeof(int));

    if (!*diag || !*upper || !*col_index || !*row_ptr)
    {
        printf("Memory allocation failed\n");
        return -1;
    }

    srand(time(NULL));

    int count = 0; // counter of non zero element
    (*row_ptr)[0] = 0;

    for (int i = 0; i < n; i++)
    {
        // diagonal is alywas full
        (*diag)[i] = (double)(rand() % 100 + 1);

        for (int j = i + 1; j < n; j++)
        {
            double r = (double)rand() / RAND_MAX; // chose if the next element will be non zero
            if (r < density)
            {
                (*upper)[count] = (double)(rand() % 100 + 1);
                (*col_index)[count] = j;
                count++;
            }
        }

        (*row_ptr)[i + 1] = count; // update next element of count
    }

    return count; // return number of non zero in the upper
}

// double get_matrix_entry_csr(){

// }

// matrix vector multiplication of a m x n matrix in CSR format
/*
   for i = 1, n
       y(i)  = 0
       for j = row_ptr(i), row_ptr(i+1) - 1
           y(i) = y(i) + val(j) * x(col_ind(j))
       end;
   end;
*/
int mv_crs_generic(double *values, // where the non-zero elements are stored, row by row
                   int *col_index, // column index of the non-zero elements
                   int *row_ptr,   // row pointer, where the first element of each row is stored
                   double *x,      // input vector (dimension = n)
                   double *y,      // output vector (dimension = m)
                   int m,          // number of rows of A
                   int n)          // number of columns of A
{
    // for every row i
    for (int i = 0; i < m; i++)
    {
        printf("row %d:\n", i);
        // take the first and last index of non-zero elements of that row in values[]
        int first = row_ptr[i];        // location index in values[] of the first non-zero element of row i
        int last = row_ptr[i + 1] - 1; // location index in values[] of the last non-zero element of row i
        // print the first and last index
        printf("Elements of row %d are stored in values[] in positions:\n ", i);
        printf("first: %d, last: %d\n", first, last);
        y[i] = 0.0;
        if (first > last)
            printf("row %d has 0 non-zero elements, we fix y[%d] = 0.0 \n", i, i);
        for (int j = first; j <= last; j++)
        {                                        // we iterate from first to last index
            y[i] += values[j] * x[col_index[j]]; // we multiply the value with the corresponding x value and we accumulate y[i]
            printf("y[%d] += %f * x[%d] = %f * %f = %f\n", i, values[j], col_index[j], values[j], x[col_index[j]], y[i]);
        }
    }

    return 0;
}

// computes matrix-vector product for symmetric matrices  in CSR format
void mv_csr_symmetric(int n,                // matrix dimension (n x n)
                      const double *diag,   // diagonal (n dense values )
                      const double *upper,  // upper non zero values, per row
                      const int *col_index, // columns index for upper values
                      const int *row_ptr,   // starting index in values for every row
                      const double *v,      // input vector
                      double *out)          // output vector
{
    // Initialize output at zero
    for (int i = 0; i < n; i++)
    {
        out[i] = 0.0;
    }

    // diagonal contribution
    for (int i = 0; i < n; i++)
    {
        out[i] += diag[i] * v[i];
    }

    // upper (and lower) triangular contribute
    for (int i = 0; i < n; i++) // for every row
    {
        int start = row_ptr[i]; // index of where the non zero elements starts in upper[] in the i-th row
        // int end = i >= (n-1) ? n : row_ptr[i + 1]; // index of where the non zero elements starts in upper[] in the i+1-th row

        int end = row_ptr[i + 1];

        // printf("upper[%d] = %f, start = %d , end = %d \n", i, upper[i], start, end);

        for (int k = start; k < end; k++) // iterate for every non zero values in i-th row
        {
            int j = col_index[k];   // save column index of non zero value A[i][j]
            double a_ij = upper[k]; // save non zero element A[i][j] value

            out[i] += a_ij * v[j]; // direct contribute of A[i][j]
            out[j] += a_ij * v[i]; // symmetric contribute (of A[j][i])
        }
    }
}

// takes the upper triangular part of a symmetric n x n matrix saved in COO
// computes the csr_row_ptr[] of CSR format
int coo_to_csr(int triangular_num_rows,  // dimension of original matrix n x n
               int upper_count,          // number of non zero elements in upper trinagular part
               double *coo_upper_values, // values saved by row in upper trinagular part
               int *coo_row_indx,        // i index array of coo
               int *coo_col_indx,        // j index array of coo
               int *csr_row_ptr)         // n + 1 size array for raw index of csr format
{
    // assert(triangular_num_rows > 0);
    // assert(upper_count > 0);

    int current_row = 0; // current row in csr_row_ptr[]
    csr_row_ptr[0] = 0;  // first row pointer is always fixed at zero

    for (int i = 0; i < upper_count; i++) // iterate for all non-zero elements
    {
        int element_row = coo_row_indx[i]; // save row index of current element
        // printf("Element visited upper[%d] = %f \n", i, coo_upper_values[i]);
        // if we encounter empty rows, we full them with the same inxed i, until we find the next row with element
        while (current_row < element_row) // if the current element is in the next row, we have to update csr_row_ptr[]
        {
            csr_row_ptr[current_row + 1] = i; // if we were in row = 2, and we found a new element in row = 3, we need to update csr_row_ptr[3]
            current_row++;                    // update index of csr_row_ptr[]

            // printf("Update to csr_row_ptr[%d] = %d \n", current_row + 1 , i);
        }
    }

    // Take care of the last empty rows
    // The last element of csr_row_ptr[triangular_dim] is alwyas = upper_count
    // If you have many last rows without elements they will be all set to upper_count
    while (current_row < triangular_num_rows)
    {
        csr_row_ptr[current_row + 1] = upper_count;
        current_row++;

        printf("Update to csr_row_ptr[%d] = %d \n", current_row + 1, upper_count);
    }

    return 0;
}

// Converts a matrix in COO format to CSR format.
// Assumes the input is only the upper triangular part of a symmetric matrix.
//
// Parameters:
//   n             - number of rows (and columns) of the matrix
//   nnz           - number of non-zero elements (in the upper triangular part)
//   values        - array of non-zero values (length = nnz)
//   row_idx       - array of row indices for each value (length = nnz)
//   col_idx       - array of column indices for each value (length = nnz)
// Output:
//   csr_row_ptr   - array of row pointers (length = n + 1)
//   csr_col_idx   - array of column indices (length = nnz)
//   csr_values    - array of values (length = nnz)
void new_coo_to_csr(int n, int nnz,
                const double *values,
                const int *row_idx,
                const int *col_idx,
                int *csr_row_ptr,
                int *csr_col_idx,
                double *csr_values)
{
    // Step 1: Initialize csr_row_ptr with zeros
    for (int i = 0; i <= n; i++)
        csr_row_ptr[i] = 0;

    // Step 2: Count how many non-zero elements are in each row
    // This is done by incrementing csr_row_ptr[row + 1] for each COO entry
    for (int k = 0; k < nnz; k++)
    {
        int row = row_idx[k];
        csr_row_ptr[row + 1]++;
    }

    // Step 3: Perform prefix sum on csr_row_ptr to get actual row start indices
    for (int i = 0; i < n; i++)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // Step 4: Fill csr_col_idx and csr_values using a temporary row pointer array
    // This temp array keeps track of the next insertion point for each row
    int *temp_row_ptr = (int *)malloc(n * sizeof(int));
    if (!temp_row_ptr)
    {
        fprintf(stderr, "Error: failed to allocate memory for temp_row_ptr\n");
        exit(EXIT_FAILURE);
    }
    memcpy(temp_row_ptr, csr_row_ptr, n * sizeof(int));

    for (int k = 0; k < nnz; k++)
    {
        int row = row_idx[k];
        int dest = temp_row_ptr[row]; // get current insert position

        csr_col_idx[dest] = col_idx[k]; // store column index
        csr_values[dest] = values[k];   // store actual value

        temp_row_ptr[row]++; // move to next insertion point for this row
    }

    // Step 5: Free temporary storage
    free(temp_row_ptr);
}

// int main()
// {
//     int n = 5;
//     double density = 0.5;
//     double *diag = NULL;
//     double *upper = NULL;
//     int *col_index = NULL;
//     int *row_ptr = NULL;

//     double v[] = {1,2,3,4,5};
//     double out[n];

//     int nnz = generate_sparse_symmetric_csr(n, density, &diag, &upper, &col_index, &row_ptr);

//     printf("diag:      ");
//     for (int i = 0; i < n; i++)
//         printf("%.1f ", diag[i]);
//     printf("\n");
//     printf("values:    ");
//     for (int i = 0; i < nnz; i++)
//         printf("%.1f ", upper[i]);
//     printf("\n");
//     printf("col_index: ");
//     for (int i = 0; i < nnz; i++)
//         printf("%d ", col_index[i]);
//     printf("\n");
//     printf("row_ptr:   ");
//     for (int i = 0; i <= n; i++)
//         printf("%d ", row_ptr[i]);
//     printf("\n");

//     printf("Matrix A (4x5):\n");
//     for (int i = 0; i < n ; i++)
//     {
//         for (int j = 0; j < n ; j++)
//         {
//             printf("%f ", get_matrix_entry_symmetric_csr(i, j, n, diag, upper, col_index, row_ptr));
//         }
//         printf("\n");
//     }

//     mv_csr_symmetric(n, diag, upper, col_index, row_ptr, v, out);
//     print_double_vector(out, n);

//     free(diag);
//     free(upper);
//     free(col_index);
//     free(row_ptr);

// return 0;

// CSR format
/*        A
    2  3  0  0
    0  0  17 2
    0  0  0  0
    0  0  0  1
*/
// double values[] = {2, 3, 17, 2, 1};
// int row_index[] = {0, 0, 1, 1, 3};
// int col_index[] = {0, 1, 2, 3, 3};
// int triag_dim = 4;
// int upper_count = 5;
// int csr_row_ptr[triag_dim + 1];
// coo_to_csr(triag_dim, upper_count, values, row_index, col_index, csr_row_ptr);

// // test upper_coo_to_csr
// // double values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
// // int col_index[] = {1, 3, 5, 2, 4, 5, 3, 5, 6, 6};
// // int row_index[] = {0, 0, 0, 1, 2, 2, 3, 4, 5, 6};
// // int triag_dim = 7;              // dimension of the upper triangular matrix
// // int upper_count = 10;           // number of non-zero elements in the upper triangular part
// // int csr_row_ptr[triag_dim + 1]; // row pointer, where the first element of each row is stored

// // free(csr_row_ptr);
// printf("CSR row pointer:\n");
// for (int i = 0; i < triag_dim + 1; i++)
// {
//     printf("%d ", csr_row_ptr[i]);
// }
// printf("\n");

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
// int nnz = 4;
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
// for(int i = 0; i < nnz; i++){
//     printf("%f ", values[i]);
// }
// printf("\n");
// // print col_index[]
// printf("col_index: ");
// for(int i = 0; i < nnz; i++){
//     printf("%d ", col_index[i]);
// }
// printf("\n");
// // print row_ptr[]
// printf("row_ptr: ");
// for(int i = 0; i < m + 1; i++){
//     printf("%d ", row_ptr[i]);
// }
// printf("\n\n");

// printf("Matrix A (4x5):\n");
// for(int i = 0; i < m; i++){
//     for(int j = 0; j < n; j++){
//         printf("%f ", get_generic_matrix_entry_csr(i, j, m, row_ptr, col_index, values));
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
