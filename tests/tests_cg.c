#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vec.h"
#include "parser.h"
#include "time.h"
#include "cg.h"
#include "coo.h"
#include "csr.h"
#include "ell.h"
#include "utils.h"

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
    int result = conjugate_gradient_coo(
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
    int result = conjugate_gradient_coo(
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
    int result = conjugate_gradient_coo(
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
void test_sparse_symmetric_random_matrix(int matrix_size, int non_zero_elements_upper)
{
    clock_t start = clock();
    printf("\n--- Test: Sparse Symmetric Random Matrix ---\n");

    int n = 2000;                  // Matrix size
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
        x[i] = 0.0;                        // Initialize x to zero
    }

    // Run the conjugate gradient method
    int result = conjugate_gradient_csr(
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

int test_cg_csr()
{
    printf("Start of program.\n");
    printf("-------------------------------------------------------------------\n");

    printf("Loading input data system from file...\n");
    const char *filename = "data/data.txt"; // linear system data input (COO) filename
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filename);
    }

    printf("Loading solution from OpenFoam for validation...\n");
    const char *filenameFoamSolution = "data/Phi"; // solution of the linear system from OpenFoam filename
    FILE *fileFoamSolution = fopen(filenameFoamSolution, "r");
    if (!filenameFoamSolution)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filenameFoamSolution);
    }

    int count_diag = 0, count_upper = 0, count_lower = 0, count_solution_openfoam = 0;

    // Parse all sections in a single file pass
    double *source = parseDoubleArray(file, "source", &count_diag);
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);

    // Parse solution from openFoam
    double *openFoamSolution = parseDoubleArraySolution(fileFoamSolution, "internalField   nonuniform List<scalar>", &count_solution_openfoam);

    // check if the number of elements in the same dimension as b (source)
    if (count_diag != count_solution_openfoam)
    {
        printf("Error: The number of elements in the solution vector does not match the number of elements in the source vector.\n");
        return -1;
    }

    // Print the number of elements in each array
    printf("\n%d\n", count_solution_openfoam);
    // Print the first few values for each array
    if (source)
    {
        printf("\n source( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf(")\n");
    }

    // Print the first few values for each array
    if (diag)
    {
        printf("\ndiag( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf(")\n");
    }

    if (upper)
    {
        printf("\nupper( \n");
        for (int i = 0; i < 5 && i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_upper - 5; i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf(")\n");

        // printf("\nupper( \n");
        // for (int i = 0; i < count_upper; i++)
        //     printf("%f ", upper[i]);
        // printf("...\n");
    }

    if (upperAddr)
    {
        printf("\nupperAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf(")\n");
    }

    if (lowerAddr)
    {
        printf("\nlowerAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf(")\n");
    }
    if (openFoamSolution)
    {
        printf("\nopenFoamSolution ( \n");
        for (int i = 0; i < 5 && i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_solution_openfoam - 5; i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf(")\n");
    }

    printf("CSR test\n\n\n");

    // int triangular_num_rows = count_diag - 1;
    // int *csr_row_ptr = (int *)malloc((count_diag + 1) * sizeof(int));
    // coo_to_csr(triangular_num_rows, count_upper, upper, upperAddr, lowerAddr, csr_row_ptr);

    int *csr_row_ptr = malloc((count_diag + 1) * sizeof(int));
    int *csr_col_idx = malloc(count_upper * sizeof(int));
    double *csr_values = malloc(count_upper * sizeof(double));

    coo_to_csr(count_diag, count_upper, upper, lowerAddr, upperAddr,
                   csr_row_ptr);

    // print_dense_symmetric_matrix_from_csr(count_diag, diag, upper, lowerAddr, csr_row_ptr);
    // compare_symmetric_matrices_coo_csr(count_diag, diag, upper, upperAddr, lowerAddr, count_upper, diag, upper, lowerAddr, csr_row_ptr);
    printf("Conversion from COO to CSR done \n csr_row_ptr[] = ");
    // print_integer_vector(csr_row_ptr, triangular_num_rows + 1);

    int n_tests = 1;
    printf("-------------------------------------------------------------------\n\n");
    for (int i = 0; i < n_tests; i++)
    {
        printf("Begin test number %d\n\n", i + 1);
        // initialize the output vector
        double *y = (double *)malloc(count_diag * sizeof(double));

        // Time the conjugate gradient method
        // Start the timer
        clock_t start = clock();
        int iterations = conjugate_gradient_csr(
            count_diag,
            diag,
            count_upper,
            csr_values,
            csr_row_ptr,
            csr_col_idx,
            source,
            y,
            1000,
            1e-6);
        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        double diff = 0;
        double max_component_diff = max_difference(y, openFoamSolution, count_diag);
        printf("Time spent: %f seconds\n", time_spent);
        if (iterations >= 0)
        {
            printf("Converged in %d iterations.\n\n", iterations);
            // print some values of x
            for (int i = 0; i < 5 && i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf("...\n");
            // Print the last few values
            for (int i = count_diag - 5; i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf(")\n");

            // Check the result against the OpenFOAM solution
            double rmse_value = rmse(y, openFoamSolution, count_diag);
            double euclidean_distance_value = euclidean_distance(y, openFoamSolution, count_diag);
            printf("RMSE: %f\n", rmse_value);
            printf("Euclidean distance: %f\n", euclidean_distance_value);
            printf("Max component difference: %f\n", max_component_diff);
        }
        else
        {
            printf("Didnt reach convergences init max_iter.\n");
        }

        free(y);
        printf("End of test number %d\n", i + 1);
        printf("-------------------------------------------------------------------\n\n");
    }
    printf("\n-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");
    printf("SUMMARY of %d runs: \n", n_tests);
    printf("-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(source);
    free(openFoamSolution);
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_values);

    fclose(file);
    fclose(fileFoamSolution);
    printf("Files closed successfully.\n");
    printf("End of program.\n");

    return 0;
}

int test_cg_ell()
{
    printf("Start of program.\n");
    printf("-------------------------------------------------------------------\n");

    printf("Loading input data system from file...\n");
    const char *filename = "data/data.txt"; // linear system data input (COO) filename
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filename);
    }

    printf("Loading solution from OpenFoam for validation...\n");
    const char *filenameFoamSolution = "data/Phi"; // solution of the linear system from OpenFoam filename
    FILE *fileFoamSolution = fopen(filenameFoamSolution, "r");
    if (!filenameFoamSolution)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filenameFoamSolution);
    }

    int count_diag = 0, count_upper = 0, count_lower = 0, count_solution_openfoam = 0;

    // Parse all sections in a single file pass
    double *source = parseDoubleArray(file, "source", &count_diag);
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);

    // Parse solution from openFoam
    double *openFoamSolution = parseDoubleArraySolution(fileFoamSolution, "internalField   nonuniform List<scalar>", &count_solution_openfoam);

    // check if the number of elements in the same dimension as b (source)
    if (count_diag != count_solution_openfoam)
    {
        printf("Error: The number of elements in the solution vector does not match the number of elements in the source vector.\n");
        return -1;
    }

    // Print the number of elements in each array
    printf("\n%d\n", count_solution_openfoam);
    // Print the first few values for each array
    if (source)
    {
        printf("\n source( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf(")\n");
    }

    // Print the first few values for each array
    if (diag)
    {
        printf("\ndiag( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf(")\n");
    }

    if (upper)
    {
        printf("\nupper( \n");
        for (int i = 0; i < 5 && i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_upper - 5; i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf(")\n");

        // printf("\nupper( \n");
        // for (int i = 0; i < count_upper; i++)
        //     printf("%f ", upper[i]);
        // printf("...\n");
    }

    if (upperAddr)
    {
        printf("\nupperAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf(")\n");
    }

    if (lowerAddr)
    {
        printf("\nlowerAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf(")\n");
    }
    if (openFoamSolution)
    {
        printf("\nopenFoamSolution ( \n");
        for (int i = 0; i < 5 && i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_solution_openfoam - 5; i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf(")\n");
    }

    printf("ELL test\n\n\n");

    // Compute the maximum number of non-zero elements in each row of the upper triangular part
    int nnz_max = compute_max_nnz_row_upper(count_diag, count_upper, lowerAddr, upperAddr);
    printf("nnz_max = %d\n", nnz_max);
    // Allocate memory for the ELL format
    double *ell_values = malloc(nnz_max * count_diag * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * count_diag * sizeof(int));

    // Convert the COO format to ELL format
    coo_to_ell_symmetric_upper(count_diag, count_upper, upper, lowerAddr, upperAddr,
                               ell_values, ell_col_idx, nnz_max);
    
                               // Print the ELL format analysis
    analyze_ell_matrix(count_diag, nnz_max, ell_values, ell_col_idx);
    
    int n_tests = 1;
    printf("-------------------------------------------------------------------\n\n");
    for (int i = 0; i < n_tests; i++)
    {
        printf("Begin test number %d\n\n", i + 1);
        // initialize the output vector
        double *y = (double *)malloc(count_diag * sizeof(double));

        // Time the conjugate gradient method
        // Start the timer
        clock_t start = clock();
        int iterations = conjugate_gradient_ell(count_diag,
                                                diag,
                                                count_upper,
                                                ell_values,
                                                ell_col_idx,
                                                nnz_max,
                                                source,
                                                y,
                                                1000,
                                                1e-6);

        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        double diff = 0;
        double max_component_diff = max_difference(y, openFoamSolution, count_diag);
        printf("Time spent: %f seconds\n", time_spent);
        if (iterations >= 0)
        {
            printf("Converged in %d iterations.\n\n", iterations);
            // print some values of x
            for (int i = 0; i < 5 && i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf("...\n");
            // Print the last few values
            for (int i = count_diag - 5; i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf(")\n");

            // Check the result against the OpenFOAM solution
            double rmse_value = rmse(y, openFoamSolution, count_diag);
            double euclidean_distance_value = euclidean_distance(y, openFoamSolution, count_diag);
            printf("RMSE: %f\n", rmse_value);
            printf("Euclidean distance: %f\n", euclidean_distance_value);
            printf("Max component difference: %f\n", max_component_diff);
        }
        else
        {
            printf("Didnt reach convergences init max_iter.\n");
        }

        free(y);
        printf("End of test number %d\n", i + 1);
        printf("-------------------------------------------------------------------\n\n");
    }
    printf("\n-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");
    printf("SUMMARY of %d runs: \n", n_tests);
    printf("-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(source);
    free(openFoamSolution);
    free(ell_values);
    free(ell_col_idx);

    fclose(file);
    fclose(fileFoamSolution);
    printf("Files closed successfully.\n");
    printf("End of program.\n");

    return 0;
}

int test_cg_ell_colmajor()
{
    printf("Start of program.\n");
    printf("-------------------------------------------------------------------\n");

    printf("Loading input data system from file...\n");
    const char *filename = "data/data.txt"; // linear system data input (COO) filename
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filename);
    }

    printf("Loading solution from OpenFoam for validation...\n");
    const char *filenameFoamSolution = "data/Phi"; // solution of the linear system from OpenFoam filename
    FILE *fileFoamSolution = fopen(filenameFoamSolution, "r");
    if (!filenameFoamSolution)
    {
        perror("Error in opening file");
        return -1;
    }
    else
    {
        printf("File %s opened successfully.\n", filenameFoamSolution);
    }

    int count_diag = 0, count_upper = 0, count_lower = 0, count_solution_openfoam = 0;

    // Parse all sections in a single file pass
    double *source = parseDoubleArray(file, "source", &count_diag);
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);

    // Parse solution from openFoam
    double *openFoamSolution = parseDoubleArraySolution(fileFoamSolution, "internalField   nonuniform List<scalar>", &count_solution_openfoam);

    // check if the number of elements in the same dimension as b (source)
    if (count_diag != count_solution_openfoam)
    {
        printf("Error: The number of elements in the solution vector does not match the number of elements in the source vector.\n");
        return -1;
    }

    // Print the number of elements in each array
    printf("\n%d\n", count_solution_openfoam);
    // Print the first few values for each array
    if (source)
    {
        printf("\n source( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("source[%d] = %f\n", i, source[i]);
        printf(")\n");
    }

    // Print the first few values for each array
    if (diag)
    {
        printf("\ndiag( \n");
        for (int i = 0; i < 5 && i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_diag - 5; i < count_diag; i++)
            printf("diag[%d] = %f\n", i, diag[i]);
        printf(")\n");
    }

    if (upper)
    {
        printf("\nupper( \n");
        for (int i = 0; i < 5 && i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_upper - 5; i < count_upper; i++)
            printf("upper[%d] = %f\n", i, upper[i]);
        printf(")\n");

        // printf("\nupper( \n");
        // for (int i = 0; i < count_upper; i++)
        //     printf("%f ", upper[i]);
        // printf("...\n");
    }

    if (upperAddr)
    {
        printf("\nupperAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("upperAddr[%d] = %d\n", i, upperAddr[i]);
        printf(")\n");
    }

    if (lowerAddr)
    {
        printf("\nlowerAddr ( \n");
        for (int i = 0; i < 5 && i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_lower - 5; i < count_lower; i++)
            printf("lowerAddr[%d] = %d\n", i, lowerAddr[i]);
        printf(")\n");
    }
    if (openFoamSolution)
    {
        printf("\nopenFoamSolution ( \n");
        for (int i = 0; i < 5 && i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf("...\n");
        // Print the last few values
        for (int i = count_solution_openfoam - 5; i < count_solution_openfoam; i++)
            printf("openFoamSolution[%d] = %f\n", i, openFoamSolution[i]);
        printf(")\n");
    }

    printf("ELL colmajor test\n\n\n");

    // Compute the maximum number of non-zero elements in each row of the upper triangular part
    int nnz_max = compute_max_nnz_row_upper(count_diag, count_upper, lowerAddr, upperAddr);
    printf("nnz_max = %d\n", nnz_max);
    // Allocate memory for the ELL format
    double *ell_values = malloc(nnz_max * count_diag * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * count_diag * sizeof(int));

    // Convert the COO format to ELL format
    // coo_to_ell_symmetric_upper(count_diag, count_upper, upper, lowerAddr, upperAddr,
    //                            ell_values, ell_col_idx, nnz_max);

    coo_to_ell_symmetric_upper_colmajor(count_diag, count_upper, upper, lowerAddr, upperAddr,
                                        ell_values, ell_col_idx, nnz_max);
    
                               // Print the ELL format analysis
    // analyze_ell_matrix(count_diag, nnz_max, ell_values, ell_col_idx);
    // analyze_ell_matrix_colmajor(count_diag, nnz_max, ell_values, ell_col_idx);
    analyze_ell_matrix_colmajor(count_diag, nnz_max, ell_values, ell_col_idx);
    
    int n_tests = 1;
    printf("-------------------------------------------------------------------\n\n");
    for (int i = 0; i < n_tests; i++)
    {
        printf("Begin test number %d\n\n", i + 1);
        // initialize the output vector
        double *y = (double *)malloc(count_diag * sizeof(double));

        // Time the conjugate gradient method
        // Start the timer
        clock_t start = clock();
        int iterations = conjugate_gradient_ell_colmajor(count_diag,
                                                diag,
                                                count_upper,
                                                ell_values,
                                                ell_col_idx,
                                                nnz_max,
                                                source,
                                                y,
                                                1000,
                                                1e-6);

        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        double diff = 0;
        double max_component_diff = max_difference(y, openFoamSolution, count_diag);
        printf("Time spent: %f seconds\n", time_spent);
        if (iterations >= 0)
        {
            printf("Converged in %d iterations.\n\n", iterations);
            // print some values of x
            for (int i = 0; i < 5 && i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf("...\n");
            // Print the last few values
            for (int i = count_diag - 5; i < count_diag; i++)
            {
                diff = y[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, y[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf(")\n");

            // Check the result against the OpenFOAM solution
            double rmse_value = rmse(y, openFoamSolution, count_diag);
            double euclidean_distance_value = euclidean_distance(y, openFoamSolution, count_diag);
            printf("RMSE: %f\n", rmse_value);
            printf("Euclidean distance: %f\n", euclidean_distance_value);
            printf("Max component difference: %f\n", max_component_diff);
        }
        else
        {
            printf("Didnt reach convergences init max_iter.\n");
        }

        free(y);
        printf("End of test number %d\n", i + 1);
        printf("-------------------------------------------------------------------\n\n");
    }
    printf("\n-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");
    printf("SUMMARY of %d runs: \n", n_tests);
    printf("-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");

    // Free allocated memory
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(source);
    free(openFoamSolution);
    free(ell_values);
    free(ell_col_idx);

    fclose(file);
    fclose(fileFoamSolution);
    printf("Files closed successfully.\n");
    printf("End of program.\n");

    return 0;
}

int main()
{

    // test_identity_matrix();

    // test_uniform_matrix();

    // // chose a matrix size and number of non-zero elements on the upper triangular part
    // int matrix_size = 2000; // Matrix size
    // int non_zero_elements_upper = matrix_size * 5; // Number of non-zero elements in the upper triangular part
    // test_sparse_symmetric_random_matrix(matrix_size, non_zero_elements_upper);

    // test_cg_csr();

    // test_cg_ell();

    test_cg_ell_colmajor();

    return 0;
}