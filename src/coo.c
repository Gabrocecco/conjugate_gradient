#include <stdio.h>
#include <stdlib.h>
#include "coo.h"
#include "csr.h"
#include <math.h>

// takes
void print_dense_symmeric_matrix_from_coo(int n, double *diag, double *upper, int *row_inx, int *col_inx, int upper_count)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double value = get_matrix_entry_symmetric_coo(i, j, n, diag, upper, row_inx, col_inx, upper_count);
            printf("%f ", value);
        }
        printf("\n");
    }
}

/*
  Generate a sparse symemtric matrix in COO 
   - n           : 
   - sparsity    : probablility of non zero in every off diagonal dimension 
  Output:
   - *upper_nnz           = 
   - *coo_values_upper    = 
   - *coo_rows, *coo_cols = indici 
   - *diagonal            = 
*/
void generate_sparse_symmetric_coo(int n, double sparsity,
                                  int *upper_nnz,
                                  double **coo_values_upper,
                                  int **coo_rows,
                                  int **coo_cols,
                                  double **diagonal)
{
    // 1) allocate and populate diagonal 
    *diagonal = malloc(n * sizeof(double));
    if (!*diagonal) { perror("malloc diagonal"); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; ++i)
        (*diagonal)[i] = rand() / (double)RAND_MAX;

    // 2) capacità massima (metà superiore completa)
    int max_entries = n*(n-1)/2;
    *coo_values_upper = malloc(max_entries * sizeof(double));
    *coo_rows        = malloc(max_entries * sizeof(int));
    *coo_cols        = malloc(max_entries * sizeof(int));
    if (!*coo_values_upper || !*coo_rows || !*coo_cols) {
        perror("malloc coo arrays");
        exit(EXIT_FAILURE);
    }

    // 3) singola passata per riempire
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            if (rand() / (double)RAND_MAX < sparsity) {
                (*coo_rows)[idx]        = i;
                (*coo_cols)[idx]        = j;
                (*coo_values_upper)[idx]= rand() / (double)RAND_MAX;
                ++idx;
            }
        }
    }

    // 4) fissiamo upper_nnz e, opzionale, riduciamo con realloc
    *upper_nnz = idx;
    if (idx < max_entries) {
        *coo_rows        = realloc(*coo_rows,        idx * sizeof(int));
        *coo_cols        = realloc(*coo_cols,        idx * sizeof(int));
        *coo_values_upper= realloc(*coo_values_upper,idx * sizeof(double));
    }
}

// this function returns the value of the matrix at (i,j) position
double get_matrix_entry_symmetric_coo(int i,
                                      int j,
                                      int n,
                                      double *diag,
                                      double *upper,
                                      int *i_indexes,
                                      int *j_indexes,
                                      int upper_count)
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

// Generate a sparse symmetric matrix in dense format with dense diagonal with a density parameter in COO format
int generate_sparse_symmetric_matrix_full_diag_coo(int n, double density, double **diag, double **upper, int **upperAddr, int **lowerAddr)
{
    // Allocate memory for the diagonal and upper triangular part
    *diag = (double *)malloc(n * sizeof(double));
    *upper = (double *)malloc(n * n * sizeof(double)); // Maximum size
    *upperAddr = (int *)malloc(n * n * sizeof(int));   // Maximum size
    *lowerAddr = (int *)malloc(n * n * sizeof(int));   // Maximum size

    if (*diag == NULL || *upper == NULL || *upperAddr == NULL || *lowerAddr == NULL)
    {
        free(*diag);
        free(*upper);
        free(*upperAddr);
        free(*lowerAddr);
        printf("Memory allocation failed\n");
        return -1;
    }

    int upper_count = 0;

    // Fill the diagonal with random values
    for (int i = 0; i < n; i++)
    {
        (*diag)[i] = (double)(rand() % 100 + 1); // Random values between 1 and 100
    }

    // Fill the upper triangular part with random values based on the density
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if ((double)rand() / RAND_MAX < density)
            {
                (*upper)[upper_count] = (double)(rand() % 100 + 1); // Random values between 1 and 100
                (*upperAddr)[upper_count] = i;
                (*lowerAddr)[upper_count] = j;
                upper_count++;
            }
        }
    }

    return upper_count; // Return the number of non-zero elements in the upper triangular part
}

int generate_sparse_generic_matrix_coo(int n, double density, double **values, int **upperAddr, int **lowerAddr)
{
    // allocate memory for the values, upperAddr and lowerAddr
    *values = (double *)malloc(n * n * sizeof(double)); // Maximum size
    *upperAddr = (int *)malloc(n * n * sizeof(int));    // Maximum size
    *lowerAddr = (int *)malloc(n * n * sizeof(int));    // Maximum size
    if (*values == NULL || *upperAddr == NULL || *lowerAddr == NULL)
    {
        free(*values);
        free(*upperAddr);
        free(*lowerAddr);
        printf("Memory allocation failed\n");
        return -1;
    }

    // Fill the matrix with random values based on the density
    int nnz_count = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if ((double)rand() / RAND_MAX < density)
            {
                (*values)[nnz_count] = (double)(rand() % 100 + 1); // Random values between 1 and 100
                (*upperAddr)[nnz_count] = i;
                (*lowerAddr)[nnz_count] = j;
                nnz_count++;
            }
        }
    }

    return nnz_count; // Return the number of non-zero elements
}
// void mv_csr_symmetric(int n,                // matrix dimension (n x n)
//                       const double *diag,   // diagonal (n dense values )
//                       const double *upper,  // upper non zero values, per row
//                       const int *col_index, // columns index for upper values
//                       const int *row_ptr,   // starting index in values for every row
//                       const double *v,      // input vector
//                       double *out)          // output vector

// COO symmetric matrix-vector multiplication
void mv_coo_symmetric(int n,               // matrix size (n x n)
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

int compare_symmetric_matrices_coo_csr(int n,
                                        double *diag_coo,
                                        double *upper_coo,
                                        int *i_idx,
                                        int *j_idx,
                                        int upper_count,
                                        double *diag_csr,
                                        double *upper_csr,
                                        int *col_index,
                                        int *row_ptr)
{
    const double TOL = 1e-6;
    int success = 1;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double a = get_matrix_entry_symmetric_coo(i, j, n, diag_coo, upper_coo, i_idx, j_idx, upper_count);
            double b = get_matrix_entry_symmetric_csr(i, j, n, diag_csr, upper_csr, col_index, row_ptr);
            double diff = fabs(a - b);
            if (diff > TOL)
            {
                printf("Mismatch at (%d, %d): COO = %.12f, CSR = %.12f, diff = %.3e\n", i, j, a, b, diff);
                success = -1;
            }
        }
    }

    if (success)
    {
        printf("Matrix comparison passed: COO == CSR within tolerance %.1e\n", TOL);
    }
    else
    {
        printf("Matrix comparison FAILED.\n");
    }

    return success;
}

// Convert COO format to CSR format
// This function converts the upper triangular part of a sparse symmetric matrix in COO format to CSR format
// int upper_coo_to_csr(int triangular_dim,       // dimension of the upper triangular matrix
//                      int upper_count,          // number of non-zero elements in the upper triangular part
//                      double *coo_upper_values, // non-zero elements in the upper triangular part, row by row
//                      int *coo_row_indx,        // i indexes of non-zero elements (coo_upper_values)
//                      int *coo_col_indx,        // j indexes of non-zero elements (coo_upper_values)
//                      int *csr_row_ptr)         // row pointer, where the first element of each row is stored
// {
//     // values[] and col_index[] are the same in CSR and COO format

//     // iterate values in col
//     csr_row_ptr[0] = 0; // first row pointer is always 0
//     int row_ptr_index = 1;
//     // iterate on all upper non zero values
//     // count the consecutive equal elements in coo_row_indx
//     int last_element = coo_row_indx[0]; // read the first element of coo_row_indx[]
//     for (int i = 1; i < upper_count; i++)
//     {
//         if (coo_row_indx[i] != last_element)
//         // if the new element if different from the old one, we must write in csr_row_ptr
//         {
//             csr_row_ptr[row_ptr_index] = i;
//             row_ptr_index++; // increment the row_ptr_index
//         }

//         last_element = coo_row_indx[i];
//     }
//     csr_row_ptr[triangular_dim] = upper_count;

//     return 0;
// }

// CSR format
/*        A
    1  0  6  0  0
    7  2  0  0  8
    0  0  3  0  0
    0  9  0  4  0
    0  10 0  0  5
*/

// CSR format
/*        A
    0  0  3  0  0
    22 0  0  0  17
    0  0  0  0  0
    0  8  0  0  0

values =    [3, 22, 17, 8]
col_index = [2, 0,  4,  1]
row_ptr =   [0, 1,  3,  3, 4]
*/

// Convert COO to CSR format
//     int coo_to_csr_symmetric_full_diag(int n, int upper_count, double *diag, double *upper, int *upperAddr, int *lowerAddr)
// {
//     // Allocate memory for CSR format
//     double *values = (double *)malloc((upper_count + n) * sizeof(double)); // Diagonal + upper values
//     int *col_index = (int *)malloc((upper_count + n) * sizeof(int));       // Column indices
//     int *row_ptr = (int *)malloc((n + 1) * sizeof(int));                   // Row pointers
//     if (values == NULL || col_index == NULL || row_ptr == NULL)
//     {
//         printf("Memory allocation failed\n");
//         return -1;
//     }
//     // Initialize row pointers

//    // iterate over the rows
//     for (int i = 0; i < n + 1; i++)
//     {
//         row_ptr[i] = 0;
//     }

//     return 0;
// }

// int main()
// {
// double *diag, *upper;
// int *upperAddr, *lowerAddr;
// int n = 10;           // Matrix size
// double density = 0.2; // Density of the matrix

// // Generate a sparse symmetric matrix of dimension n with a density
// int upper_count = generate_sparse_symmetric_matrix_full_diag_coo(n, density, &diag, &upper, &upperAddr, &lowerAddr);

// // Print the generated matrix
// printf("Diagonal: ");
// for (int i = 0; i < n; i++)
// {
//     printf("%f ", diag[i]);
// }
// printf("\n");

// printf("Upper: ");
// for (int i = 0; i < upper_count; i++)
// {
//     printf("%f ", upper[i]);
// }
// printf("\n");

// printf("UpperAddr: ");
// for (int i = 0; i < upper_count; i++)
// {
//     printf("%d ", upperAddr[i]);
// }
// printf("\n");

// printf("LowerAddr: ");
// for (int i = 0; i < upper_count; i++)
// {
//     printf("%d ", lowerAddr[i]);
// }
// printf("\n");

// // print the matrix in dense format
// printf("Matrix in dense format:\n");
// for (int i = 0; i < n; i++)
// {
//     for (int j = 0; j < n; j++)
//     {
//         double value = get_matrix_entry_symmetric_coo(i, j, n, diag, upper, upperAddr, lowerAddr, upper_count);
//         printf("%f ", value);
//     }
//     printf("\n");
// }

// // Free allocated memory
// free(diag);
// free(upper);
// free(upperAddr);
// free(lowerAddr);

// CSR format
/*        A
    1  0  6  0  0
    0  2  0  0  8
    6  0  3  0  0
    0  0  0  4  0
    0  8  0  0  5
*/
// double values[] = {1.0, 6.0, 2.0, 8.0, 6.0, 3.0, 4.0, 8.0, 5.0};
// int col_index[] = {0, 2, 1, 4, 0, 2, 3, 1, 4};
// int row_ptr[] = {0, 2, 4, 6, 7, 9}; // lunghezza = n_righe + 1 = 6

// // print the matrix
// printf("Matrix A (CSR format):\n");
// printf("Values: ");
// for (int i = 0; i < 9; i++)
// {
//     printf("%f ", values[i]);
// }
// printf("\n");
// printf("Column indices: ");
// for (int i = 0; i < 9; i++)
// {
//     printf("%d ", col_index[i]);
// }
// printf("\n");
// printf("Row pointers: ");
// for (int i = 0; i < 6; i++)
// {
//     printf("%d ", row_ptr[i]);
// }
// printf("\n\n");

// for (int i = 0; i < 5; i++)
// {
//     for (int j = 0; j < 5; j++)
//     {
//         printf("%f ", get_matrix_entry_csr(i, j, row_ptr, col_index, values));
//     }
//     printf("\n");
// }

//     return 0;
// }