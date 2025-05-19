#include "vec.h"
#include <math.h>

// COO matrix-vector multiplication
// void mv_coo(int n,               // matrix size (n x n)
//             int coo_length,      // number of non-zero elements in the upper triangular part
//             const double *diag,  // diagonal elements (exactly n dense elements)
//             const double *value, // non-zero elements in the upper triangular part
//             const int *rows,     // i indexes of non-zero elements (value)
//             const int *columns,  // j indexes of non-zero elements (value)
//             const double *v,     // input vector
//             double *out)
// { // output vector

//     // init output vector
//     for (int i = 0; i < n; i++)
//     {
//         out[i] = 0.0;
//     }

//     // dense diagonal
//     for (int i = 0; i < n; i++)
//     {
//         out[i] += diag[i] * v[i]; // accumulate the contribution of the diagonal
//     }

//     // upper triangular matrix
//     for (int element_index = 0; element_index < coo_length; element_index++)
//     {                                                    // iterate over all non-zero elements in values (top triangular part, by row)
//         const int row_index = rows[element_index];       // i, actual row index of A
//         const int column_index = columns[element_index]; // j, actual column index of A
//         const double val = value[element_index];         // A[i][j]
//         out[row_index] += val * v[column_index];         // out[i] += A[i][j] * v[j]
//         out[column_index] += val * v[row_index];         // out[j] += A[j][i] * v[i]   // symmetric contribution
//     }
// }

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

double vec_l1norm(double *a, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++)
    {
        result += fabs(a[i]);
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

// compare two vectors and return the root mean square error
double rmse(const double *a, const double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum / n);   // return sqrt(1/n *sum(diff^2))
}

// compare two vectors and return the euclidean distance
double euclidean_distance(const double *a, const double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// compare two vectors and return the maximum difference
double max_difference(const double *a, const double *b, int n) {
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}