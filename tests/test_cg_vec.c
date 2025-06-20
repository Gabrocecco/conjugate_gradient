#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vec.h"
#include "parser.h"
#include "time.h"
#include "coo.h"
#include "csr.h"
#include "ell.h"
#include "utils.h"
#include "cg_vec.h"
#include "vectorized.h"

int test_cg_ell_full_colmajor_vectorized()
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

    printf("ELL full colmajor test\n\n\n");

    // Compute the maximum number of non-zero elements in each row of the upper triangular part
    int nnz_max = compute_max_nnz_row_full(count_diag, count_upper, lowerAddr, upperAddr);
    printf("nnz_max = %d\n", nnz_max);

    // Allocate memory for the ELL format
    double *ell_values = malloc(nnz_max * count_diag * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * count_diag * sizeof(int));
    // uint64_t *ell_col_idx = malloc(nnz_max * count_diag * sizeof(uint64_t));

    // Convert the COO format to ELL format
    coo_to_ell_symmetric_full_colmajor(count_diag, count_upper, upper, lowerAddr, upperAddr, ell_values, ell_col_idx, nnz_max);

    // coo_to_ell_symmetric_full_colmajor_sdtint(count_diag, count_upper, upper, lowerAddr, upperAddr, ell_values, ell_col_idx, nnz_max)

    // Print the ELL format analysis
    analyze_ell_matrix_full_colmajor(count_diag, nnz_max, ell_values, ell_col_idx);

    uint64_t *ell_cols64 =
        malloc(nnz_max * count_diag * sizeof(uint64_t));
    for (int k = 0; k < nnz_max * count_diag; ++k)
    {
        ell_cols64[k] = (uint64_t)ell_col_idx[k];
    }

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
        int iterations = conjugate_gradient_ell_full_colmajor_vectorized(
            count_diag,  // n
            diag,        // diag[]
            count_upper, // upper_count
            ell_values,  // valori ELL
            ell_cols64,  // indici ELL (uint64_t*)
            nnz_max,     // max_nnz_row
            source,      // b[]
            y,           // x[] (output)
            1000,        // max_iter
            1e-6         // tol
        );

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

int main(void)
{
    return test_cg_ell_full_colmajor_vectorized();
}