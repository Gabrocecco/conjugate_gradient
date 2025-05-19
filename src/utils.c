#include <stdio.h>
#include "utils.h"

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
