#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ell.h"

/*
    MatVec prodcut for generic matrix A (nxx) in ELL format.    TO-DO: test this function
*/

void print_dense_matrix(double **A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%6.1f ", A[i][j]);
        printf("\n");
    }
}

void convert_ell_to_dense(int n, double *diag, double *ell_values, int *ell_cols, int max_nnz_row, double **A)
{
    for (int i = 0; i < n; i++)
    {
        A[i][i] = diag[i];
        for (int k = 0; k < max_nnz_row; k++)
        {
            int col = ell_cols[i * max_nnz_row + k];
            double val = ell_values[i * max_nnz_row + k];
            if (col != -1)
            {
                A[i][col] = val;
                A[col][i] = val; // symmetric
            }
        }
    }
}

void convert_ell_to_dense_colmajor(int n, double *diag, double *ell_values, int *ell_cols, int max_nnz_row, double **A)
{
    for (int i = 0; i < n; i++)
    {
        A[i][i] = diag[i];
        for (int k = 0; k < max_nnz_row; k++)
        {
            int col = ell_cols[k * n + i];
            double val = ell_values[k * n + i];
            if (col != -1)
            {
                A[i][col] = val;
                A[col][i] = val; // symmetric
            }
        }
    }
}

/*
    Convert full-ELL (row-major) → dense
    - n              : dimensione matrice
    - diag[i]        : diagonale A[i][i]
    - ell_values     : off-diagonali, size = n*max_nnz_row, layout row-major (i*max_nnz_row + k)
    - ell_cols       : colonne corrispondenti, -1 per padding
    - max_nnz_row    : numero slot per riga
    - A              : puntatore a vettore di puntatori (A[i][j])
*/
void convert_ell_full_to_dense_rowmajor(int n,
                                        const double diag[],
                                        const double ell_values[],
                                        const int ell_cols[],
                                        int max_nnz_row,
                                        double **A)
{
    // 0) azzera tutta la matrice
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = 0.0;

    // 1) inserisci la diagonale
    for (int i = 0; i < n; ++i)
        A[i][i] = diag[i];

    // 2) inserisci ogni slot off-diagonale
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < max_nnz_row; ++k)
        {
            int idx = i * max_nnz_row + k;
            int col = ell_cols[idx];
            if (col == -1)
                continue; // padding → skip

            double v = ell_values[idx];
            A[i][col] = v; // A[i][j]
            A[col][i] = v; // simmetrico A[j][i]
        }
    }
}

/*
    Convert full-ELL (column-major) -> dense
    - ell_values / ell_cols layout: slot-major (k*n + i)
*/
void convert_ell_full_to_dense_colmajor(int n,
                                        const double diag[],
                                        const double ell_values[],
                                        const int ell_cols[],
                                        int max_nnz_row,
                                        double **A)
{
    // 0) azzera tutta la matrice
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = 0.0;

    // 1) inserisci la diagonale
    for (int i = 0; i < n; ++i)
        A[i][i] = diag[i];

    // 2) inserisci ogni slot off-diagonale
    for (int k = 0; k < max_nnz_row; ++k)
    {
        for (int i = 0; i < n; ++i)
        {
            int idx = k * n + i;
            int col = ell_cols[idx];
            if (col == -1)
                continue; // padding → skip

            double v = ell_values[idx];
            A[i][col] = v; // A[i][j]
            A[col][i] = v; // simmetrico A[j][i]
        }
    }
}

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
        for (int j = 0; j < max_nnz_row; ++j) // possible loop unrolling, max_nnz_row is small and constant
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
    Compute the maximum number of non-zero elements in each row of a full symmetric matrix
    in COO format, considering both upper and lower triangular parts.
    This function assumes that the input is a full symmetric matrix, so it counts both
    upper and lower triangular elements.
*/
int compute_max_nnz_row_full(int n,
                             int upper_nnz,
                             const int *coo_rows,
                             const int *coo_cols)
{
    int *row_counts = calloc(n, sizeof(int));
    if (!row_counts)
        return -1;

    // iterate over the upper triangular elements in COO format
    for (int k = 0; k < upper_nnz; ++k)
    {
        // Get the row and column indices of the current element
        int i = coo_rows[k];
        int j = coo_cols[k];

        // Check if the element is in the upper triangular part
        if (j <= i)
        {
            fprintf(stderr, "Error: non-upper entry row=%d, col=%d\n", i, j);
            free(row_counts);
            return -1;
        }

        // Increment the count for the row and its symmetric counterpart
        row_counts[i]++;
        row_counts[j]++;
    }

    int max = 0;
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

/*
    Analyze ELL (column-major) for symmetric matrix:
    - n        = number of rows
    - nnz_max  = max non-zeros per row (number of “slots”)
    - ell_values[idx]  stores the value at slot j, row i via idx = j*n + i
    - ell_col_idx[idx] stores the corresponding column index (or –1 if padding)
*/
void analyze_ell_matrix_colmajor(int n, int nnz_max,
                                 const double *ell_values,
                                 const int *ell_col_idx)
{
    int padding_count = 0;
    int mismatch_count = 0;
    int count_rows_with_1 = 0;
    int count_rows_with_2 = 0;
    int count_rows_with_3 = 0;
    int count_rows_with_4 = 0;

    // --- Print values row by row ---
    printf("ELL values (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            double v = ell_values[idx];
            printf("%8.2f ", v);
            if (v == 0.0)
            {
                padding_count++;
                if (ell_col_idx[idx] != -1)
                {
                    mismatch_count++;
                }
            }
        }
        printf("\n");
    }

    // --- Print column-indices row by row ---
    printf("\nELL column indices (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            printf("%4d ", ell_col_idx[idx]);
        }
        printf("\n");
    }

    // --- General statistics ---
    int total_slots = n * nnz_max;
    double padding_pct = 100.0 * padding_count / total_slots;
    double utilization = 100.0 * (total_slots - padding_count) / total_slots;

    printf("\nTotal slots: %d\n", total_slots);
    printf("Padding count: %d\n", padding_count);
    printf("Padding percentage: %.2f%%\n", padding_pct);
    printf("Memory utilization: %.2f%%\n", utilization);
    if (mismatch_count == 0)
        printf("All zero values correctly marked with column index -1.\n");
    else
        printf("Warning: %d padding entries have col_idx != -1.\n", mismatch_count);

    // --- Count rows by number of real non-zero entries ---
    for (int i = 0; i < n; ++i)
    {
        int real_count = 0;
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            if (ell_values[idx] != 0.0)
                real_count++;
        }
        if (real_count == 1)
            count_rows_with_1++;
        else if (real_count == 2)
            count_rows_with_2++;
        else if (real_count == 3)
            count_rows_with_3++;
        else if (real_count == 4)
            count_rows_with_4++;
    }

    double pct1 = 100.0 * count_rows_with_1 / n;
    double pct2 = 100.0 * count_rows_with_2 / n;
    double pct3 = 100.0 * count_rows_with_3 / n;
    double pct4 = 100.0 * count_rows_with_4 / n;

    printf("\nRows with exactly 1 non-zero: %d of %d (%.2f%%)\n",
           count_rows_with_1, n, pct1);
    printf("Rows with exactly 2 non-zeros: %d of %d (%.2f%%)\n",
           count_rows_with_2, n, pct2);
    printf("Rows with exactly 3 non-zeros: %d of %d (%.2f%%)\n",
           count_rows_with_3, n, pct3);
    printf("Rows with exactly 4 non-zeros: %d of %d (%.2f%%)\n",
           count_rows_with_4, n, pct4);
}

void analyze_ell_matrix_full_colmajor(int n, int nnz_max,
                                      const double *ell_values,
                                      const int *ell_col_idx)
{
    int padding_count = 0;
    int mismatch_count = 0;

    // --- Statistiche per conteggio righe per numero di non-zeri reali ---
    int *row_nnz_counts = (int *)calloc(nnz_max + 1, sizeof(int));

    // --- Print valori ---
    printf("ELL-FULL values (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            double v = ell_values[idx];
            printf("%8.2f ", v);
            if (v == 0.0)
            {
                padding_count++;
                if (ell_col_idx[idx] != -1)
                    mismatch_count++;
            }
        }
        printf("\n");
    }

    // --- Print colonne ---
    printf("\nELL-FULL column indices (row × slot):\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            printf("%4d ", ell_col_idx[idx]);
        }
        printf("\n");
    }

    // --- Statistiche generali ---
    int total_slots = n * nnz_max;
    double padding_pct = 100.0 * padding_count / total_slots;
    double utilization = 100.0 * (total_slots - padding_count) / total_slots;

    printf("\nTotal slots: %d\n", total_slots);
    printf("Padding count: %d\n", padding_count);
    printf("Padding percentage: %.2f%%\n", padding_pct);
    printf("Memory utilization: %.2f%%\n", utilization);
    if (mismatch_count == 0)
        printf("All zero values correctly marked with column index -1.\n");
    else
        printf("Warning: %d padding entries have col_idx != -1.\n", mismatch_count);

    // --- Conta righe per numero di non-zeri reali ---
    for (int i = 0; i < n; ++i)
    {
        int real_count = 0;
        for (int j = 0; j < nnz_max; ++j)
        {
            int idx = j * n + i;
            if (ell_values[idx] != 0.0)
                real_count++;
        }
        if (real_count <= nnz_max)
            row_nnz_counts[real_count]++;
    }

    // --- Stampa distribuzione ---
    printf("\nRow non-zero statistics:\n");
    for (int k = nnz_max; k >= 0; --k)
    {
        if (row_nnz_counts[k] > 0)
        {
            double pct = 100.0 * row_nnz_counts[k] / n;
            printf("Rows with exactly %d non-zeros: %d of %d (%.2f%%)\n",
                   k, row_nnz_counts[k], n, pct);
        }
    }

    free(row_nnz_counts);
}


/*
    COO -> ELL conversion for symmetric matrix: upper triangular part only
    Stored in COLUMN-MAJOR order.
*/
void coo_to_ell_symmetric_upper_colmajor(int n,                    // dimension of matrix A (n x n)
                                         int upper_nnz,            // number of non-zeros in upper triangular part
                                         double *coo_values_upper, // values of upper triangular part
                                         int *coo_rows,            // row indices of upper triangular part
                                         int *coo_cols,            // column indices of upper triangular part
                                         double *ell_values,       // ELL values array (to be filled; size = n * max_nnz_row)
                                         int *ell_cols,            // ELL column indices array (to be filled; same size)
                                         int max_nnz_row           // maximum number of non-zeros per row in ELL format
)
{
    // 1) allocate current_count array to keep track of how many elements we have already inserted in each row
    int *current_count = (int *)calloc(n, sizeof(int));
    if (!current_count)
    {
        fprintf(stderr, "Error: calloc failed\n");
        return;
    }

    // 2) initialize ELL arrays
    // ELL values at 0
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));
    // ELL column indices to -1 (indicating no entry)
    for (int k = 0; k < n * max_nnz_row; ++k)
    {
        ell_cols[k] = -1;
    }

    // 3) save in ELL arrays every COO entry
    // iterate over all the upper triangular non zeros
    for (int k = 0; k < upper_nnz; ++k)
    {
        // for each element, save row, col and values of the current upper triangular element from COO format
        int row = coo_rows[k];
        int col = coo_cols[k];
        double val = coo_values_upper[k];

        // verify that the element is in the upper triangular part
        if (col <= row)
        {
            printf("Error: Found non-upper element in COO: row=%d, col=%d\n", row, col);
            free(current_count);
            return;
        }

        // slot is the column index of ELL columns for the current element
        int slot = current_count[row];
        current_count[row]++; // next time we will insert a value from this row, we will use the next slot

        // check if we have exceeded the maximum number of non-zeros per row
        if (slot >= max_nnz_row)
        {
            printf("Error: row %d has more than %d non-zeros\n", row, max_nnz_row);
            free(current_count);
            return;
        }

        // we are in slot column of ELL, each slot is exactly n elements, in this slot we are on the row-th element
        ell_values[slot * n + row] = val;
        ell_cols[slot * n + row] = col;
    }

    free(current_count);
}

/*
    COO -> ELL conversion for symmetric matrix: full (upper+lower) off-diagonals,
    diagonal kept external. Stored in COLUMN-MAJOR order.
    Inputs:
      - n                : dimension of A (n×n)
      - upper_nnz        : number of COO entries (strictly upper triangular)
      - coo_values_upper : values of strictly upper part
      - coo_rows, coo_cols: row/col indices of strictly upper part
      - max_nnz_row      : maximum non-zeros per row in the full matrix (off-diag only)
    Outputs (size = n * max_nnz_row):
      - ell_values
      - ell_cols         : column indices, -1 for padding
*/
void coo_to_ell_symmetric_full_colmajor(int n,                      
                                        int upper_nnz,
                                        double *coo_values_upper,
                                        int *coo_rows,
                                        int *coo_cols,
                                        double *ell_values,
                                        int *ell_cols,
                                        int max_nnz_row)
{
    // 1) allocate and zero count of entries per row
    int *current_count = (int *)calloc(n, sizeof(int));
    if (!current_count)
    {
        fprintf(stderr, "Error: calloc failed\n");
        return;
    }

    // 2) init ELL arrays: values=0, cols=-1
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));

    for (int k = 0; k < n * max_nnz_row; ++k)
        ell_cols[k] = -1;

    // 3) scatter each upper-offdiag entry twice: (row -> col) and (col -> row)
    for (int k = 0; k < upper_nnz; ++k)
    {
        // saves indexes and value of the current upper triangular element from COO format
        int i = coo_rows[k];
        int j = coo_cols[k];
        double v = coo_values_upper[k];

        // sanity check (check if the element is in the upper triangular part)
        if (j <= i)
        {
            printf("Error: non-upper COO entry row=%d, col=%d\n", i, j);
            free(current_count);
            return;
        }

        // -- place A[i][j] into row i --
        int slot_i = current_count[i]++; // get the current slot for row i and increment it

        // check if we have exceeded the maximum number of non-zeros per row
        if (slot_i >= max_nnz_row)
        {
            printf("Error: row %d has >%d off-diag entries\n", i, max_nnz_row);
            free(current_count);
            return;
        }

        // column-major index for slot_i, row i
        ell_values[slot_i * n + i] = v;
        ell_cols[slot_i * n + i] = j;

        // -- place A[j][i] into row j (symmetric) --
        int slot_j = current_count[j]++;
        if (slot_j >= max_nnz_row)
        {
            printf("Error: row %d has >%d off-diag entries\n", j, max_nnz_row);
            free(current_count);
            return;
        }
        ell_values[slot_j * n + j] = v;
        ell_cols[slot_j * n + j] = i;
    }

    free(current_count);
}

/*
    COO -> ELL conversion for symmetric matrix: full (upper+lower) off-diagonals,
    diagonal kept external. Stored in COLUMN-MAJOR order.
    Inputs:
      - n                : dimension of A (n×n)
      - upper_nnz        : number of COO entries (strictly upper triangular)
      - coo_values_upper : values of strictly upper part
      - coo_rows, coo_cols: row/col indices of strictly upper part
      - max_nnz_row      : maximum non-zeros per row in the full matrix (off-diag only)
    Outputs (size = n * max_nnz_row):
      - ell_values
      - ell_cols         : column indices, -1 for padding
*/
void coo_to_ell_symmetric_full_colmajor_sdtint(int n,
                                               int upper_nnz,
                                               double *coo_values_upper,
                                               int *coo_rows,
                                               int *coo_cols,
                                               double *ell_values,
                                               uint64_t *ell_cols,
                                               int max_nnz_row)
{
    // 1) allocate and zero count of entries per row
    int *current_count = (int *)calloc(n, sizeof(int));
    if (!current_count)
    {
        fprintf(stderr, "Error: calloc failed\n");
        return;
    }

    // 2) init ELL arrays: values=0, cols=-1
    memset(ell_values, 0, n * max_nnz_row * sizeof(double));

    for (int k = 0; k < n * max_nnz_row; ++k)
        ell_cols[k] = (uint64_t)-1;

    // 3) scatter each upper-offdiag entry twice: (row→col) and (col→row)
    for (int k = 0; k < upper_nnz; ++k)
    {
        // saves indexes and value of the current upper triangular element from COO format
        int i = coo_rows[k];
        int j = coo_cols[k];
        double v = coo_values_upper[k];

        // sanity check (check if the element is in the upper triangular part)
        if (j <= i)
        {
            printf("Error: non-upper COO entry row=%d, col=%d\n", i, j);
            free(current_count);
            return;
        }

        // -- place A[i][j] into row i --
        int slot_i = current_count[i]++; // get the current slot for row i and increment it

        // check if we have exceeded the maximum number of non-zeros per row
        if (slot_i >= max_nnz_row)
        {
            printf("Error: row %d has >%d off-diag entries\n", i, max_nnz_row);
            free(current_count);
            return;
        }

        // column-major index for slot_i, row i
        ell_values[slot_i * n + i] = v;
        ell_cols[slot_i * n + i] = (uint64_t)j;

        // -- place A[j][i] into row j (symmetric) --
        int slot_j = current_count[j]++;
        if (slot_j >= max_nnz_row)
        {
            printf("Error: row %d has >%d off-diag entries\n", j, max_nnz_row);
            free(current_count);
            return;
        }
        ell_values[slot_j * n + j] = v;
        ell_cols[slot_j * n + j] = (uint64_t)i;
    }

    free(current_count);
}

/*
    MatVec for symmetric matrix in ELL format (upper triangular only + full diagonal stored separately)
    Stored in COLUMN-MAJOR order.
*/
void mv_ell_symmetric_upper_colmajor(int n,              // dimension of matrix A (n x n)
                                     int max_nnz_row,    // maximum number of non-zeros per row in ELL format
                                     double *diag,       // diagonal elements (size n)
                                     double *ell_values, // ELL values array (size n * max_nnz_row)
                                     int *ell_cols,      // ELL column indices array (size n * max_nnz_row)
                                     double *x,          // input vector (size n)
                                     double *y           // output vector (size n)
)
{
    // 1) initialize y = 0
    for (int i = 0; i < n; ++i)
    {
        y[i] = 0.0;
    }

    // 2) diagonal contribution
    for (int i = 0; i < n; ++i)
    {
        // usare +=
        y[i] = y[i] + diag[i] * x[i]; // ! to vectorize , vector vector element wise multiplication and accumulate (y = y + x * z)
    }

    // ! Per vettorizzare, usare matrice mergiata unica

    // 3) for each "slot" j in ELL (column-major)
    for (int j = 0; j < max_nnz_row; ++j)
    {
        for (int i = 0; i < n; ++i) // iterate over rows (elements on the j-th slot)
        {
            /* Each slot has exactly n elements. We are in the j-th slot at the i-th element.*/
            int offset = j * n + i; // compute offset via offset
            /* ell_values and ell_cols have the same layout. */
            int col = ell_cols[offset]; // extract column offset of the current element

            if (col == -1)
            {
                // padding entry, skip (accumulate zero)
                continue;
            }
            if (col <= i)
            {
                // non-upper entry found -> error
                printf("Error: non-upper element at row=%d, col=%d\n", i, col);
                return;
            }

            double v = ell_values[offset]; // extract value of the current element
            // accumulate contribution for row i
            y[i] = y[i] + v * x[col]; // ! axpy = y = y + alpha * x
            // symmetric contribution for row col
            y[col] = y[col] + v * x[i]; // ! axpy = y = y + alpha * x
        }
    }
}

/*
    MatVec for generic matrix with external diag, in col major order.
*/
void mv_ell_symmetric_full_colmajor(int n,
                                    int max_nnz_row, // max number of off-diagonal nnz in rows
                                    double *diag,
                                    double *ell_values, // ELL values (size n * max_nnz_row)
                                    int *ell_cols,      // ELL column indices (size n * max_nnz_row)
                                    double *x,          // input vector
                                    double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    // 2) diagonal contributes (element wise product + accumulate)
    for (int i = 0; i < n; ++i)
    {
        y[i] += diag[i] * x[i];
    }

    // 3) non-diagonal contributes
    // iterate all slots (ELL columns)
    for (int j = 0; j < max_nnz_row; ++j)
    {
        // iterate all elements in a slot
        for (int i = 0; i < n; ++i)
        {
            int offset = j * n + i; // column-major offset

            int col = ell_cols[offset];

            // do not accumulate if we are on a 0 padded element
            if (col == -1) // padding
                continue;

            // off-diagonal contribute
            y[i] += ell_values[offset] * x[col];
        }
    }
}

/*
    MatVec for generic matrix with external diag, in col major order.
*/
void mv_ell_symmetric_full_colmajor_sdtint(int n,
                                           int max_nnz_row, // max number of off-diagonal nnz in rows
                                           double *diag,
                                           double *ell_values, // ELL values (size n * max_nnz_row)
                                           uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                           double *x,          // input vector
                                           double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    // 2) diagonal contributes (element wise product + accumulate)
    for (int i = 0; i < n; ++i)
    {
        y[i] += diag[i] * x[i];
    }

    // 3) non-diagonal contributes
    // iterate all slots (ELL columns)
    for (int j = 0; j < max_nnz_row; ++j)
    {
        // iterate all elements in a slot
        for (int i = 0; i < n; ++i)
        {
            int offset = j * n + i; // column-major offset

            int col = ell_cols[offset];

            // do not accumulate if we are on a 0 padded element
            if (col == -1) // padding
                continue;

            // off-diagonal contribute
            y[i] += ell_values[offset] * x[col];
        }
    }
}
