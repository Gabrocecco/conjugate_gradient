#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ell.h"

/*
    From COO format to ELL format for symmetric matrix, saving all elements.    TO-DO: test this function
*/
void coo_to_ell_symmetric_full(int n,              // dimension of matrix A (n x n)
                               double *diag,       // diagonal values of A
                               int upper_nnz,      // number of non-zeros in upper triangular part
                               double *coo_values, // values of upper triangular part
                               int *coo_rows,      // row indices of upper triangular part
                               int *coo_cols,      // column indices of upper triangular part
                               double *ell_values, // ELL values array (to be filled)
                               int *ell_cols       // ELL column indices array (to be filled)
)
{

    // Determine maximum number of non-zeros per row (including diagonal and symmetric elements)
    int *row_counts = (int *)calloc(n, sizeof(int));

    for (int i = 0; i < n; ++i)
        row_counts[i]++; // diagonal element

    for (int i = 0; i < upper_nnz; ++i)
    {
        int row = coo_rows[i];
        int col = coo_cols[i];
        row_counts[row]++;
        row_counts[col]++;
    }

    int max_nnz_row = 0;
    for (int i = 0; i < n; ++i)
    {
        if (row_counts[i] > max_nnz_row)
            max_nnz_row = row_counts[i];
    }

    // Initialize ELLPACK arrays
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));
    for (int i = 0; i < n * max_nnz_row; ++i)
        ell_cols[i] = -1;

    int *current_count = (int *)calloc(n, sizeof(int));

    // Insert diagonal elements
    for (int row = 0; row < n; ++row)
    {
        int ell_offset = row * max_nnz_row;
        ell_values[ell_offset] = diag[row];
        ell_cols[ell_offset] = row;
        current_count[row] = 1;
    }

    // Insert upper and symmetric lower elements
    for (int i = 0; i < upper_nnz; ++i)
    {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double val = coo_values[i];

        // upper triangular insertion
        int ell_offset_row = row * max_nnz_row + current_count[row];
        ell_values[ell_offset_row] = val;
        ell_cols[ell_offset_row] = col;
        current_count[row]++;

        // symmetric lower triangular insertion
        int ell_offset_col = col * max_nnz_row + current_count[col];
        ell_values[ell_offset_col] = val;
        ell_cols[ell_offset_col] = row;
        current_count[col]++;
    }

    free(row_counts);
    free(current_count);
}

/* 
    MatVec prodcut for generic matrix A (nxx) in ELL format.    TO-DO: test this function
*/

void mv_ell(int n,              // dimension of matrix A (n x n)
            int max_nnz_row,    // maximum number of non-zeros per row in ELL format
            double *ell_values, // ELL values array (size n * max_nnz_row)
            int *ell_cols,      // ELL column indices array (size n * max_nnz_row)
            double *x,          // input vector (size n)
            double *y)          // output vector (size n)
{
    // Iterate for every element of output vector y
    for (int i = 0; i < n; ++i)
    {
        y[i] = 0.0; // initialize output vector element

        // Iterate over the element of every ELL row
        for (int j = 0; j < max_nnz_row; ++j)
        {
            int col = ell_cols[i * max_nnz_row + j]; // extract column index of element

            if (col >= 0)   // check if the column index is valid (no padding)
            {                                                     
                y[i] += ell_values[i * max_nnz_row + j] * x[col]; // accumulate the product
            }
        }
    }
}

/* 
    MatVec for symmetric matrix in ELL format (upper triangular only + full diagonal stored separately)
    This function computes the matrix-vector product y = A * x,
    where A is a symmetric matrix stored in ELL format with upper triangular part only.
    The diagonal elements are stored separately.
    The input vector x and output vector y are of size n.
*/
void mv_ell_symmetric_upper(int n,              // dimension of matrix A (n x n)
                            int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                            double *diag,       // diagonal elements (size n)
                            double *ell_values, // upper triangular values only (no diagonal)
                            int *ell_cols,      // column indices (same layout as ell_values)
                            double *x,          // input vector (size n)
                            double *y           // output vector (size n)
)
{
    // Initialize output to zero
    for (int i = 0; i < n; ++i)
        y[i] = 0.0;

    // iterate for every elements of output vector y
    for (int i = 0; i < n; ++i)
    {
        y[i] += diag[i] * x[i]; // contribution from the diagonal

        // for each row of ELL, iterate over the upper triangular elements
        for (int j = 0; j < max_nnz_row; ++j)
        {
            int col = ell_cols[i * max_nnz_row + j]; // extract column index of element

            if (col == -1)
                continue; // skip padding entries

            if (col > i) // ensure upper triangle only (excluding diagonal)
            {
                double val = ell_values[i * max_nnz_row + j]; // extract value of element
                y[i] += val * x[col];                         // contribution from upper triangular part
                y[col] += val * x[i];                         // symmetric contribution
            }
            else // non-upper triangulat element found, we can exit
            {
                printf("Error: Found a non-upper triangular element in ELL format: row=%d, col=%d\n", i, col);
                return; // exit if we find a non-upper triangular element
            }
        }
    }
}

/*
    COO -> ELL conversion for symmetric matrix: upper triangular part only (diagonal separate)
    1) Allocate ELL arrays with size n * max_nnz_row with all elements initialized to zero.
    2) Initialize column indices to -1 (indicating no entry).
    3) Iterate over the upper triangular elements in COO format:
    a) For each element, check if it is in the upper triangular part (col > row).
    b) If it is, insert the value into the ELL values array and update the column index.
    c) If a non-upper triangular element is found, print an error message and exit.
*/
void coo_to_ell_symmetric_upper(int n,                    // dimension of matrix A (n x n)
                                int upper_nnz,            // number of non-zeros in upper triangular part
                                double *coo_values_upper, // values of upper triangular part
                                int *coo_rows,            // row indices of upper triangular part
                                int *coo_cols,            // column indices of upper triangular part
                                double *ell_values,       // ELL values array (to be filled)
                                int *ell_cols,            // ELL column indices array (to be filled)
                                int max_nnz_row           // maximum number of non-zeros per row in ELL format
)
{
    // Initialize ELL arrays, exatly n * max_nnz_row elements
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));

    // initialize column indices to -1 (indicating no entry)
    for (int i = 0; i < n * max_nnz_row; ++i)
        ell_cols[i] = -1;

    // current_count[] will keep track of how many elements we have already inserted in each row
    int *current_count = (int *)calloc(n, sizeof(int));

    // Fill ELL structures with only the upper triangular part
    for (int i = 0; i < upper_nnz; ++i) // iterate over the upper triangular elements
    {
        // save row, col and values of the current upper triangular element using COO format
        int row = coo_rows[i];
        int col = coo_cols[i];
        double val = coo_values_upper[i];

        if (col > row) // only consider upper triangular elements
        {
            int offset = row * max_nnz_row + current_count[row]; // compute offset of this element in the ELL array, each row has exactly max_nnz_row elements, and may have padding
            ell_values[offset] = val;
            ell_cols[offset] = col;
            current_count[row]++; // update the count of elements in this row
        }
        else // if we find a non-upper triangular element, we can exit
        {
            printf("Error: Found a non-upper triangular element in COO format: row=%d, col=%d\n", row, col);
            free(current_count);

            return; // exit if we find a non-upper triangular element
        }
    }

    free(current_count);

    return;
}

/* 
    Compute the maximum number of non-zero elements in each row of the upper triangular part
    of a symmetric matrix in COO format.
*/
int compute_max_nnz_row_upper(int n, int upper_nnz, int *coo_rows, int *coo_cols)
{
    int *row_counts = calloc(n, sizeof(int));
    if (!row_counts)
        return -1;

    for (int i = 0; i < upper_nnz; ++i)
    {
        int row = coo_rows[i];
        int col = coo_cols[i];
        if (col > row)
            row_counts[row]++;
        else
        {
            printf("Error: Found a non-upper triangular element in COO format: row=%d, col=%d\n", row, col);
            free(row_counts);
            return -1;
        }
    }

    int max = 0;
    for (int i = 0; i < n; ++i)
        if (row_counts[i] > max)
            max = row_counts[i];

    free(row_counts);

    return max;
}