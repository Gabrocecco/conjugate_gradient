
#include <stdlib.h>
#include <string.h>

// Function to get the value of the matrix at (i, j) position
double get_matrix_entry(int i,
                        int j,
                        int n,
                        double *diag,
                        double *upper,
                        int *i_indexes,
                        int *j_indexes,
                        int upper_count);

// COO matrix-vector multiplication
void mv_coo(int n,               // matrix size (n x n)
            int coo_length,      // number of non-zero elements in the upper triangular part
            const double *diag,  // diagonal elements (exactly n dense elements)
            const double *value, // non-zero elements in the upper triangular part
            const int *rows,     // i indexes of non-zero elements (value)
            const int *columns,  // j indexes of non-zero elements (value)
            const double *v,     // input vector
            double *out);        // output vector

// Vector subtraction: out = a - b
void vec_sub(double *a, double *b, double *out, int n);

// Vector addition: out = a + b
void vec_add(double *a, double *b, double *out, int n);

// Vector assignment: a = b
void vec_assign(double *a, double *b, int n);

// Dot product: out = a^T * b
double vec_dot(double *a, double *b, int n);

// AXPY operation: out = a + alpha * b
void vec_axpy(double *a, double *b, double alpha, double *out, int n);

double rmse(const double *a, const double *b, int n);

double euclidean_distance(const double *a, const double *b, int n);

double max_difference(const double *a, const double *b, int n);

double vec_l1norm(double *a, int n);

