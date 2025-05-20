#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "parser.h"
#include "vec.h"
#include "tests_cg.h"
#include "cg.h"
#include "csr.h"
#include "utils.h"
#include "coo.h"

/*
    Compile:
        gcc -std=c99 -Wall -pedantic -o main main.c tests.c parser.c vec.c -lm -g -fsanitize=address
        gcc -std=c99 -Wall -pedantic -o main main.c tests.c parser.c vec.c -lm -O3 -DNDEBUG -march=native -mtune=native -v
    Run:
        ./main
    source          2000 ( 0.000000 0.000000 0.000000 0.000000 ...);
    diag            2000 ( -3.29086 -4.0237 -4.00375 -4.01 ...);
    upper           3890 ( 0.864978 1.21294 1.21294 0.943508, ...);
    upperAddr       3890 ( 1 10 190 2 11 191 3, ...);
    lowerAddr       3890 ( 0 0 0 1 1 1 2 2 2 3 3 3 4 4, ...);

*/

int main()
{
    printf("Start of program.\n");
    printf("-------------------------------------------------------------------\n");

    printf("Loading input data system from file...\n");
    const char *filename = "data/data.txt";  // linear system data input (COO) filename 
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
    const char *filenameFoamSolution = "data/Phi";   // solution of the linear system from OpenFoam filename 
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
    
    //check if the number of elements in the same dimension as b (source)
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

    // ---------------------------------------------------------------------------------------------------------------------------------------------- //
    int n_tests = 1;
    double time_spent_acc = 0;
    printf("-------------------------------------------------------------------\n\n");
    for (int i = 0; i < n_tests; i++)
    {
        printf("Begin test number %d\n\n", i + 1);
        // initialize the output vector
        double *x = (double *)malloc(count_diag * sizeof(double));

        // Time the conjugate gradient method
        // Start the timer
        clock_t start = clock();
        int iterations = conjugate_gradient_coo(
            count_diag,  // matrix size (n x n)
            diag,        // diagonal elements (exactly n dense elements)
            count_upper, // number of non-zero elements in the upper triangular part
            upper,       // non-zero elements in the upper triangular part
            upperAddr,   // i indexes of non-zero elements (upper)
            lowerAddr,   // j indexes of non-zero elements (upper)
            source,      // input vector
            x,           // output vector
            1000,        // maximum number of iterations
            1e-6);       // tolerance for convergence
        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        time_spent_acc += time_spent;
        double diff = 0;
        double max_component_diff = max_difference(x, openFoamSolution, count_diag);
        printf("Time spent: %f seconds\n", time_spent);
        if (iterations >= 0)
        {
            printf("Converged in %d iterations.\n\n", iterations);
            // print some values of x
            for (int i = 0; i < 5 && i < count_diag; i++)
            {
                diff = x[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, x[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf("...\n");
            // Print the last few values
            for (int i = count_diag - 5; i < count_diag; i++)
            {
                diff = x[i] - openFoamSolution[i];
                printf("x[%d] = %.10f | OpenFoam: x[%d] = %.10f | difference: %f | %.3f%%\n", i, x[i], i, openFoamSolution[i], diff, diff * 100);
            }
            printf(")\n");

            // Check the result against the OpenFOAM solution
            double rmse_value = rmse(x, openFoamSolution, count_diag);
            double euclidean_distance_value = euclidean_distance(x, openFoamSolution, count_diag);
            printf("RMSE: %f\n", rmse_value);
            printf("Euclidean distance: %f\n", euclidean_distance_value);
            printf("Max component difference: %f\n", max_component_diff);
        }
        else
        {
            printf("Didnt reach convergences init max_iter.\n");
        }

        free(x);
        printf("End of test number %d\n", i + 1);
        printf("-------------------------------------------------------------------\n\n");
    }
    printf("\n-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");
    printf("SUMMARY of %d runs: \n", n_tests);
    printf("Average time spent in %d runs: %f seconds\n", n_tests, time_spent_acc / n_tests);
    printf("-------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------\n");

    printf("CSR test\n\n\n");

    int *csr_row_ptr = (int *)malloc((count_diag + 1) * sizeof(int));
    coo_to_csr(count_diag, count_upper, upper, upperAddr, lowerAddr, csr_row_ptr);
    // print_dense_symmetric_matrix_from_csr(count_diag, diag, upper, lowerAddr, csr_row_ptr);
    // compare_symmetric_matrices_coo_csr(count_diag, diag, upper, upperAddr, lowerAddr, count_upper, diag, upper, lowerAddr, csr_row_ptr);
    printf("Conversion from COO to CSR done \n csr_row_ptr[] = ");
    print_integer_vector(csr_row_ptr, count_diag + 1);

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
            upper,
            csr_row_ptr,
            lowerAddr,
            source,
            y,
            1000,
            1e-6);
        // Stop the timer
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        time_spent_acc += time_spent;
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
    printf("Average time spent in %d runs: %f seconds\n", n_tests, time_spent_acc / n_tests);
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

    
    fclose(file);
    fclose(fileFoamSolution);
    printf("Files closed successfully.\n");
    printf("End of program.\n");

    return 0;
}