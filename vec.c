#include "vec.h"

// this function returns the value of the matrix at (i,j) position
double get_matrix_entry(int i,
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

// COO matrix-vector multiplication
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

// out = a - b
void vec_sub(double *a, double *b, double *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] - b[i];
    }
}

// out = a + b
void vec_add(double *a, double *b, double *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

// out = a = b
void vec_assign(double *a, double *b, int n)
{
    // for (int i = 0; i < n; i++)
    // {
    //     a[i] = b[i];
    // }
    memcpy(a, b, n * sizeof(double));
}

// out = a^T * b
double vec_dot(double *a, double *b, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

// out = a + alpha * b
void vec_axpy(double *a, double *b, double alpha, double *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] + alpha * b[i];
    }
}