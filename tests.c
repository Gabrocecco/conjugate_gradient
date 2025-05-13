#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vec.h"
#include "parser.h"
#include "main.h" // Include the declaration of conjugate_gradient
#include "time.h"


// Test CG with a zero matrix
void test_zero_matrix()
{   
    clock_t start = clock();
    printf("\n--- Test: Zero Matrix ---\n");

    int n = 2000;                             // Dimensione della matrice
    double *diag = calloc(n, sizeof(double)); // Diagonale nulla
    double *upper = NULL;                     // Nessun elemento nella parte superiore
    int *rows = NULL;                         // Nessun indice di riga
    int *cols = NULL;                         // Nessun indice di colonna
    double *b = malloc(n * sizeof(double));   // Vettore dei termini noti
    double *x = malloc(n * sizeof(double));   // Vettore soluzione

    // Inizializza b con valori casuali e x a zero
    for (int i = 0; i < n; i++)
    {
        b[i] = (double)(i + 1); // Ad esempio, b = [1, 2, 3, ..., n]
        x[i] = 0.0;
    }

    // Esegui il metodo del gradiente coniugato
    int result = conjugate_gradient(
        n,     // Dimensione della matrice
        diag,  // Diagonale nulla
        0,     // Nessun elemento nella parte superiore
        upper, // Nessun elemento nella parte superiore
        rows,  // Nessun indice di riga
        cols,  // Nessun indice di colonna
        b,     // Vettore dei termini noti
        x,     // Vettore soluzione
        1000,  // Numero massimo di iterazioni
        1e-6   // Tolleranza
    );

    // Verifica il risultato
    if (result >= 0)
    {
        printf("Converged in %d iterations.\n", result);
    }
    else
    {
        printf("Failed to converge.\n");
    }

    // print 5 values of x
    for (int i = 0; i < 5 && i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf("...\n");
    // Print the last few values
    for (int i = n - 5; i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf(")\n");

    // Libera la memoria allocata
    free(diag);
    free(b);
    free(x);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f seconds\n", time_spent);
    printf("--- End of Test: Zero Matrix ---\n");
    
}

// Test CG with an identity matrix
void test_identity_matrix()
{
    clock_t start = clock();
    printf("\n--- Test: Identity Matrix ---\n");

    int n = 2000;                              // Dimensione della matrice
    double *diag = malloc(n * sizeof(double)); // Diagonale della matrice identità
    double *upper = NULL;                      // Nessun elemento nella parte superiore
    int *rows = NULL;                          // Nessun indice di riga
    int *cols = NULL;                          // Nessun indice di colonna
    double *b = malloc(n * sizeof(double));    // Vettore dei termini noti
    double *x = malloc(n * sizeof(double));    // Vettore soluzione

    // Inizializza la diagonale con 1 (matrice identità)
    for (int i = 0; i < n; i++)
    {
        diag[i] = 1.0;
    }

    // Inizializza b con valori casuali e x a zero
    for (int i = 0; i < n; i++)
    {
        b[i] = (double)(i + 1); // Ad esempio, b = [1, 2, 3, ..., n]
        x[i] = 0.0;
    }

    // Esegui il metodo del gradiente coniugato
    int result = conjugate_gradient(
        n,     // Dimensione della matrice
        diag,  // Diagonale della matrice identità
        0,     // Nessun elemento nella parte superiore
        upper, // Nessun elemento nella parte superiore
        rows,  // Nessun indice di riga
        cols,  // Nessun indice di colonna
        b,     // Vettore dei termini noti
        x,     // Vettore soluzione
        1000,  // Numero massimo di iterazioni
        1e-6   // Tolleranza
    );

    // Verifica il risultato
    if (result >= 0)
    {
        printf("Converged in %d iterations.\n", result);
    }
    else
    {
        printf("Failed to converge.\n");
    }

    // print 5 values of x
    for (int i = 0; i < 5 && i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf("...\n");
    // Print the last few values
    for (int i = n - 5; i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf(")\n");

    // Libera la memoria allocata
    free(diag);
    free(b);
    free(x);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f seconds\n", time_spent);
    printf("--- End of Test: Identity Matrix ---\n");
}

// Test CG with a matrix where every element is 1/n
void test_uniform_matrix()
{
    clock_t start = clock();
    printf("\n--- Test: Uniform Matrix (1/n) ---\n");

    int n = 2000; // Dimensione della matrice
    double one_over_n = 1.0 / n;

    // Alloca memoria per la matrice e i vettori
    double *diag = malloc(n * sizeof(double));                  // Diagonale
    double *upper = malloc((n * (n - 1) / 2) * sizeof(double)); // Parte superiore
    int *upperAddr = malloc((n * (n - 1) / 2) * sizeof(int));   // Indici riga
    int *lowerAddr = malloc((n * (n - 1) / 2) * sizeof(int));   // Indici colonna
    double *b = malloc(n * sizeof(double));                     // Vettore dei termini noti
    double *x = malloc(n * sizeof(double));                     // Vettore soluzione

    // Inizializza la diagonale con 1/n
    for (int i = 0; i < n; i++)
    {
        diag[i] = one_over_n;
    }

    // Inizializza la parte superiore con 1/n
    int index = 0;
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            upper[index] = one_over_n;
            upperAddr[index] = i;
            lowerAddr[index] = j;
            index++;
        }
    }

    // Inizializza il vettore b con tutti 1
    for (int i = 0; i < n; i++)
    {
        b[i] = 1.0;
        x[i] = 0.0; // Inizializza x a zero
    }

    // Esegui il metodo del gradiente coniugato
    int result = conjugate_gradient(
        n,         // Dimensione della matrice
        diag,      // Diagonale
        index,     // Numero di elementi non nulli nella parte superiore
        upper,     // Parte superiore
        upperAddr, // Indici riga
        lowerAddr, // Indici colonna
        b,         // Vettore dei termini noti
        x,         // Vettore soluzione
        1000,      // Numero massimo di iterazioni
        1e-6       // Tolleranza
    );

    // Verifica il risultato
    if (result >= 0)
    {
        printf("Converged in %d iterations.\n", result);
    }
    else
    {
        printf("Failed to converge.\n");
    }

    // print 5 values of x
    for (int i = 0; i < 5 && i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf("...\n");
    // Print the last few values
    for (int i = n - 5; i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf(")\n");

    // Libera la memoria allocata
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(b);
    free(x);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f seconds\n", time_spent);
    printf("--- End of Test: Uniform Matrix (1/n) ---\n");
}

#include <time.h> // For random number generation

// Test CG with a sparse symmetric matrix with random values
void test_sparse_symmetric_random_matrix()
{
    clock_t start = clock();
    printf("\n--- Test: Sparse Symmetric Random Matrix ---\n");

    int n = 2000; // Matrix size
    int non_zero_elements = n * 3; // Approximate number of non-zero elements
    printf("Matrix size: %d x %d\n", n, n);
    printf("Number of non-zero elements: %d\n", non_zero_elements);
    printf("Density: %.2f%%\n", (double)non_zero_elements / (n * n) * 100);

    // Allocate memory for the matrix and vectors
    double *diag = malloc(n * sizeof(double));                  // Diagonal
    double *upper = malloc(non_zero_elements * sizeof(double)); // Upper triangular part
    int *upperAddr = malloc(non_zero_elements * sizeof(int));   // Row indices
    int *lowerAddr = malloc(non_zero_elements * sizeof(int));   // Column indices
    double *b = malloc(n * sizeof(double));                     // Right-hand side vector
    double *x = malloc(n * sizeof(double));                     // Solution vector

    // Seed the random number generator
    srand(time(NULL));

    // Initialize the diagonal with random positive values
    for (int i = 0; i < n; i++)
    {
        diag[i] = (double)(rand() % 100 + 1); // Random values between 1 and 100
    }

    // Initialize the upper triangular part with random values
    int index = 0;
    for (int i = 0; i < n - 1 && index < non_zero_elements; i++)
    {
        for (int j = i + 1; j < n && index < non_zero_elements; j++)
        {
            if (rand() % 10 < 3) // 30% chance of being a non-zero element
            {
                double value = (double)(rand() % 100 + 1); // Random values between 1 and 100
                upper[index] = value;
                upperAddr[index] = i;
                lowerAddr[index] = j;
                index++;
            }
        }
    }

    // Initialize the vector b with random values
    for (int i = 0; i < n; i++)
    {
        b[i] = (double)(rand() % 100 + 1); // Random values between 1 and 100
        x[i] = 0.0; // Initialize x to zero
    }

    // Run the conjugate gradient method
    int result = conjugate_gradient(
        n,         // Matrix size
        diag,      // Diagonal
        index,     // Number of non-zero elements in the upper triangular part
        upper,     // Upper triangular part
        upperAddr, // Row indices
        lowerAddr, // Column indices
        b,         // Right-hand side vector
        x,         // Solution vector
        1000,      // Maximum number of iterations
        1e-6       // Tolerance
    );

    // Check the result
    if (result >= 0)
    {
        printf("Converged in %d iterations.\n", result);
    }
    else
    {
        printf("Failed to converge.\n");
    }

    // Print 5 values of x
    for (int i = 0; i < 5 && i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf("...\n");
    // Print the last few values
    for (int i = n - 5; i < n; i++)
    {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
    printf(")\n");

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(b);
    free(x);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f seconds\n", time_spent);
    printf("--- End of Test: Sparse Symmetric Random Matrix ---\n");
}


// int main(){

//     test_identity_matrix();

//     test_uniform_matrix();

//     return 0;
// }