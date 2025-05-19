#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "csr.h"
#include "utils.h"

// Test 1: verify CSR generation 
void test_generate_sparse_symmetric_csr()
{
    int n = 5;
    double density = 0.4;
    double *diag, *upper;
    int *col_index, *row_ptr;

    int upper_count = generate_sparse_symmetric_csr(n, density, &diag, &upper, &col_index, &row_ptr);

    assert(diag != NULL);
    assert(upper != NULL);
    assert(col_index != NULL);
    assert(row_ptr != NULL);
    assert(upper_count >= 0);

    printf("Test 1 (Generation of CSR matrix): PASSED\n");
    printf("Matrix in dense format:\n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, row_ptr);

    printf("diag: \n"); 
    print_double_vector(diag, n);
    printf("upper: \n");
    print_double_vector(upper, upper_count);
    printf("row_idx: \n");
    print_integer_vector(row_ptr, n + 1);
    printf("col idx: ");
    print_integer_vector(col_index, upper_count);

    free(diag);
    free(upper);
    free(col_index);
    free(row_ptr);
}


// Test 2: verify that get_matrix_entry_symmetric_csr return correct values
void test_get_matrix_entry_symmetric_csr()
{
    int n = 4;
    double diag[] = {10, 20, 30, 40};
    double upper[] = {1, 2, 3};
    int col_index[] = {1, 2, 3};
    int row_ptr[] = {0, 1, 2, 3, 3}; 

    printf("Dense matrix: \n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, row_ptr);

    // Diagonale
    assert(get_matrix_entry_symmetric_csr(0, 0, n, diag, upper, col_index, row_ptr) == 10);
    assert(get_matrix_entry_symmetric_csr(3, 3, n, diag, upper, col_index, row_ptr) == 40);

    // Triangolare superiore
    assert(get_matrix_entry_symmetric_csr(0, 1, n, diag, upper, col_index, row_ptr) == 1);
    assert(get_matrix_entry_symmetric_csr(1, 2, n, diag, upper, col_index, row_ptr) == 2);

    // Triangolare inferiore (simmetrico)
    assert(get_matrix_entry_symmetric_csr(2, 1, n, diag, upper, col_index, row_ptr) == 2);

    // Elementi nulli
    assert(get_matrix_entry_symmetric_csr(3, 2, n, diag, upper, col_index, row_ptr) == 3);

    printf("Test 2 (test_get_matrix_entry_symmetric_csr ): PASSED\n");
}

// Test 3: matrix-vector product 
void test_mv_csr_symmetric()
{
    int n = 3;
    double diag[] = {2, 3, 4};
    double upper[] = {1, 2, 3};  // A[0][1], A[0][2], A[1][2]
    int col_index[] = {1, 2, 2};
    int row_ptr[] = {0, 2, 3, 3};

    double x[] = {1.0, 2.0, 3.0};
    double y[3];
    printf("Matrix in dense format:\n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, row_ptr);

    printf("Vector:\n");
    print_double_vector(x, 3);
    mv_csr_symmetric(n, diag, upper, col_index, row_ptr, x, y);

    // Calcolo atteso:
    // y[0] = 2*1 + 1*2 + 2*3 = 2 + 2 + 6 = 10
    // y[1] = 3*2 + 1*1 + 3*3 = 6 + 1 + 9 = 16
    // y[2] = 4*3 + 2*1 + 3*2 = 12 + 2 + 6 = 20

    assert(fabs(y[0] - 10.0) < 1e-6);
    assert(fabs(y[1] - 16.0) < 1e-6);
    assert(fabs(y[2] - 20.0) < 1e-6);

    printf("Test 3 (Matrix-vector product CSR symemrtic): PASSED\n");
}

int main()
{
    test_generate_sparse_symmetric_csr();
    test_get_matrix_entry_symmetric_csr();
    test_mv_csr_symmetric();

    printf("All CSR tests completed with success!\n");
    return 0;
}
