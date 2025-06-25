#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "vec.h"
#include "parser.h"
#include "coo.h"
#include "csr.h"
#include "ell.h"
#include "utils.h"
#include "cg_vec.h"
#include "vectorized.h"
#include "common.h"
#include "cg.h"

int test_cg_ell_full_colmajor_vectorized()
{
    printf("Start of program.\n");
    const char *filename = "data/data.txt";
    const char *filenameFoamSolution = "data/Phi";

    FILE *file = fopen(filename, "r");
    FILE *fileFoamSolution = fopen(filenameFoamSolution, "r");

    if (!file || !fileFoamSolution) {
        perror("Error opening input files");
        return -1;
    }

    int count_diag = 0, count_upper = 0, count_lower = 0, count_solution_openfoam = 0;

    double *source = parseDoubleArray(file, "source", &count_diag);
    double *diag = parseDoubleArray(file, "diag", &count_diag);
    double *upper = parseDoubleArray(file, "upper", &count_upper);
    int *upperAddr = parseIntArray(file, "upperAddr", &count_upper);
    int *lowerAddr = parseIntArray(file, "lowerAddr", &count_lower);
    double *openFoamSolution = parseDoubleArraySolution(fileFoamSolution, "internalField   nonuniform List<scalar>", &count_solution_openfoam);

    if (count_diag != count_solution_openfoam) {
        fprintf(stderr, "Error: mismatched dimensions.\n");
        return -1;
    }

    int nnz_max = compute_max_nnz_row_full(count_diag, count_upper, lowerAddr, upperAddr);
    printf("nnz_max = %d\n", nnz_max);

    double *ell_values = malloc(nnz_max * count_diag * sizeof(double));
    int *ell_col_idx = malloc(nnz_max * count_diag * sizeof(int));
    coo_to_ell_symmetric_full_colmajor(count_diag, count_upper, upper, lowerAddr, upperAddr, ell_values, ell_col_idx, nnz_max);

    uint64_t *ell_cols64 = malloc(nnz_max * count_diag * sizeof(uint64_t));
    for (int k = 0; k < nnz_max * count_diag; ++k)
        ell_cols64[k] = (uint64_t)ell_col_idx[k];

    int n_tests = 1;
    struct timespec start, end;
    double time_scalar = 0.0, time_vector = 0.0;

    printf("\n=== Running %d test(s) ===\n\n", n_tests);
    for (int i = 0; i < n_tests; i++) {
        printf("Test #%d\n", i + 1);

        double *y_scalar = malloc(count_diag * sizeof(double));
        double *y_vector = malloc(count_diag * sizeof(double));

        // --- Scalar CG ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        int iterations_scalar = conjugate_gradient_ell_full_colmajor(
            count_diag, diag, count_upper, ell_values, ell_col_idx,
            nnz_max, source, y_scalar, 1000, 1e-6);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_scalar = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

        // --- Vectorized CG ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        int iterations_vector = conjugate_gradient_ell_full_colmajor_vectorized(
            count_diag, diag, count_upper, ell_values, ell_cols64,
            nnz_max, source, y_vector, 1000, 1e-6);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_vector = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

        printf("Time scalar     : %.6f s\n", time_scalar);
        printf("Time vectorized : %.6f s\n", time_vector);
        printf("Speedup (time)  : %.2fx\n", time_scalar / time_vector);

        int pass = 1;
        for (int j = 0; j < count_diag; j++) {
            if (!fp_eq(y_scalar[j], y_vector[j], 1e-6)) {
                pass = 0;
                break;
            }
        }
        printf("%s\n", pass ? "PASS: Results match!" : "FAIL: Results do NOT match!");

        double rmse_value = rmse(y_vector, openFoamSolution, count_diag);
        printf("RMSE vs OpenFOAM: %f\n", rmse_value);

        free(y_scalar);
        free(y_vector);
        printf("End of Test #%d\n\n", i + 1);
    }

    // Cleanup
    free(diag);
    free(upper);
    free(upperAddr);
    free(lowerAddr);
    free(source);
    free(openFoamSolution);
    free(ell_values);
    free(ell_col_idx);
    free(ell_cols64);
    fclose(file);
    fclose(fileFoamSolution);

    return 0;
}

int main(void) {
    return test_cg_ell_full_colmajor_vectorized();
}
