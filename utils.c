#include<stdio.h>
#include"utils.h"

// takes 
void print_dense_matrix_from_coo();

void print_dense_matrix_from_csr();

void print_integer_vector(int *vector, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", vector[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}

void print_double_vector(double *vector, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%.6f", vector[i]); 
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}


