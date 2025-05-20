#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <stdlib.h>

void print_integer_vector(int *vector, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        printf("%d", vector[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}

void print_double_vector(double *vector, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        printf("%.6f", vector[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}

int compare_double_vectors_fixed_tol(const double *a, const double *b, int n)
{
    const double TOL_ABS = 1e-6; // tolleranza fissa

    for (int i = 0; i < n; i++)
    {
        double diff = fabs(a[i] - b[i]);
        if (diff > TOL_ABS) {
            printf("Mismatch at index %d: a = %.12f, b = %.12f, diff = %.12e\n", i, a[i], b[i], diff);
            return -1;
        }
    }
    printf("Vectors are equal: \n");
    return 0;
}

int compare_int_vectors_fixed_tol(const int *a, const int *b, int n)
{
    const int TOL = 0; // tolleranza fissa (0 = confronto esatto)

    for (int i = 0; i < n; i++)
    {
        int diff = abs(a[i] - b[i]);
        if (diff > TOL) {
            printf("Mismatch at index %d: a = %d, b = %d, diff = %d\n", i, a[i], b[i], diff);
            return -1;
        }
    }
    printf("Vectors are equal: \n");
    return 1;
}
