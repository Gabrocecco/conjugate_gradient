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

            if (col >= 0) // check if the column index is valid (no padding)
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
                            double *ell_values, // upper triangular values only, with padding (size n * max_nnz_row)
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
            /* We are in the i-th row of A, every row of ELL has exactly max_nnz_row values, and in the current row we are on the j-th element*/
            int offset = i * max_nnz_row + j; // compute offset in the ELL arrays

            int col = ell_cols[offset]; // extract column index of cuurent element suing offset
            /* ell_values and ell_cols have the same layout, we can use the same offset per referencig the same element. */

            if (col == -1)
                continue; // skip padding entries

            if (col > i) // check if the element is in the upper triangular part
            {
                double val = ell_values[offset]; // extract value of element
                y[i] += val * x[col];            // contribution from upper triangular part
                y[col] += val * x[i];            // symmetric contribution
            }
            else // non-upper triangulat element found, we can exit
            {
                printf("Error: Found a non-upper triangular element in ELL format: row=%d, col=%d\n", i, col);
                return; // exit if we find a non-upper triangular element
            }
        }
    }
}

void mv_ell_symmetric_upper_opt(int n,              // dimension of matrix A (n x n)
                                int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                                double *diag,       // diagonal elements (size n)
                                double *ell_values, // upper triangular values only, with padding (size n * max_nnz_row)
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

        int base = i * max_nnz_row; // base offset for the current row in ELL arrays

        // for each row of ELL, iterate over the upper triangular elements
        for (int j = 0; j < max_nnz_row; ++j)   // ! possible loop unrolling, max_nnz_row is small and constant
        {
            /* We are in the i-th row of A, every row of ELL has exactly max_nnz_row values, and in the current row we are on the j-th element*/
            int offset = base + j; // compute offset in the ELL arrays

            int col = ell_cols[offset]; // extract column index of cuurent element suing offset
            /* ell_values and ell_cols have the same layout, we can use the same offset per referencig the same element. */

            if (col == -1)
                continue; // skip padding entries

            // assume col > i, since we are in the upper triangular part
            double val = ell_values[offset]; // extract value of element
            y[i] += val * x[col];            // contribution from upper triangular part
            y[col] += val * x[i];            // symmetric contribution
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
    // Initialize ELL array, exatly n * max_nnz_row elements, ( n rows, each with max_nnz_row elements )
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));

    // initialize all column indices to -1 (indicating no entry)
    for (int i = 0; i < n * max_nnz_row; ++i)
        ell_cols[i] = -1;

    // current_count[] will keep track of how many elements we have already inserted in each row
    int *current_count = (int *)calloc(n, sizeof(int));

    // iterate over all the upper triangular non zeros
    for (int i = 0; i < upper_nnz; ++i)
    {
        // save row, col and values of the current upper triangular element from COO format
        int row = coo_rows[i];
        int col = coo_cols[i];
        double val = coo_values_upper[i];

        if (col > row) // check if the element is int the upper traingular part
        {
            int offset = row * max_nnz_row + current_count[row]; // compute offset of this element in the ELL array, each row has exactly max_nnz_row elements, and may have padding
            ell_values[offset] = val;                            // save the element value in the correct position in the ELL value array
            ell_cols[offset] = col;                              // with the same offset, save the column index of this element in the ELL column index array
            current_count[row]++;                                // update the count of elements in this row
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
int compute_max_nnz_row_upper(int n,         // dimension of matrix A (n x n)
                              int upper_nnz, // number of non-zeros in upper triangular part
                              int *coo_rows, // row indices of upper triangular part
                              int *coo_cols  // column indices of upper triangular part
)
{
    // Allocate an array to count non-zero elements in each row
    int *row_counts = calloc(n, sizeof(int));
    if (!row_counts)
        return -1;

    // Iterate over the upper triangular elements in COO format
    for (int i = 0; i < upper_nnz; ++i)
    {
        // Get the row and column indices of the current element
        int row = coo_rows[i];
        int col = coo_cols[i];

        if (col > row)         // only consider upper triangular elements
            row_counts[row]++; // increment the count for the row
        else                   // if we find a non-upper triangular element, we can exit
        {
            printf("Error: Found a non-upper triangular element in COO format: row=%d, col=%d\n", row, col);
            free(row_counts);
            return -1;
        }
    }

    int max = 0;
    // Find the maximum count across all rows
    for (int i = 0; i < n; ++i)
        if (row_counts[i] > max)
            max = row_counts[i];

    free(row_counts);

    return max;
}

/*
    Analyze the ELL matrix structure, printing values, column indices, padding statistics.
*/
void analyze_ell_matrix(int n, int nnz_max, double *ell_values, int *ell_col_idx)
{
    int padding_count = 0;
    int mismatch_count = 0;

    printf("ELL matrix values (ell_values):\n");
    for (int i = 0; i < n * nnz_max; i++)
    {
        if (i % nnz_max == 0 && i != 0)
            printf("\n");

        printf("%8.2f ", ell_values[i]);

        if (ell_values[i] == 0.0)
            padding_count++;

        // Consistency check: padding value should have column index -1
        if (ell_values[i] == 0.0 && ell_col_idx[i] != -1)
        {
            mismatch_count++;
        }
    }
    printf("\n");

    printf("\nELL column indices (ell_col_idx):\n");
    for (int i = 0; i < n * nnz_max; i++)
    {
        if (i % nnz_max == 0 && i != 0)
            printf("\n");

        printf("%4d ", ell_col_idx[i]);
    }
    printf("\n");

    // Print global padding statistics
    int total_slots = n * nnz_max;
    double padding_percentage = 100.0 * padding_count / total_slots;
    double utilization = 100.0 * (total_slots - padding_count) / total_slots;

    printf("\nTotal slots: %d\n", total_slots);
    printf("Padding count: %d\n", padding_count);
    printf("Padding percentage: %.2f%%\n", padding_percentage);
    printf("Memory utilization: %.2f%%\n", utilization);

    if (mismatch_count == 0)
        printf("All zero values correctly marked with column index -1.\n");
    else
        printf("Warning: %d padding values have col_idx != -1.\n", mismatch_count);

    // // Per-row analysis
    // printf("\nPer-row analysis:\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     int row_real = 0, row_padding = 0;
    //     for (int j = 0; j < nnz_max; ++j)
    //     {
    //         int idx = i * nnz_max + j;
    //         if (ell_values[idx] == 0.0)
    //             row_padding++;
    //         else
    //             row_real++;
    //     }
    //     printf("Row %3d: real values = %2d, padding = %2d\n", i, row_real, row_padding);
    // }

    // Find and print rows without padding
    int full_rows_count = 0;

    printf("\nRows without padding (completely full rows):\n");

    for (int i = 0; i < n; ++i)
    {
        int has_padding = 0;
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = i * nnz_max + j;
            if (ell_values[idx] == 0.0)
            {
                has_padding = 1;
                break;
            }
        }
        if (!has_padding)
        {
            // printf("Row %d\n", i);
            full_rows_count++;
        }
    }

    printf("Total full rows without padding: %d out of %d\n", full_rows_count, n);
    // print percentage of full rows
    double percentage_full_rows = 100.0 * full_rows_count / n;
    printf("Percentage of full rows without padding: %.2f%%\n", percentage_full_rows);

    // --- Find and print rows with exactly 3 real values ---
    int count_rows_with_3 = 0;

    printf("\nRows with exactly 3 non-zero values:\n");

    for (int i = 0; i < n; ++i)
    {
        int real_count = 0;
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = i * nnz_max + j;
            if (ell_values[idx] != 0.0)
                real_count++;
        }
        if (real_count == 3)
        {
            // printf("Row %d\n", i);
            count_rows_with_3++;
        }
    }

    printf("Total rows with exactly 3 non-zero values: %d of %d\n", count_rows_with_3, n);
    // print percentage of rows with exactly 3 non-zero values
    double percentage_rows_with_3 = 100.0 * count_rows_with_3 / n;
    printf("Percentage of rows with exactly 3 non-zero values: %.2f%%\n", percentage_rows_with_3);
}
