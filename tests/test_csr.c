#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "csr.h"
#include "coo.h"
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

    // printf("Dense matrix: \n");
    // print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, row_ptr);

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

// void test_mv_csr_symmetric_null_row()
// {
//     int n = 3;
//     double diag[] = {1, 2, 3, 4};
//     double upper[] = {5, 6, 7}; 
//     int col_index[] = {2, 3, 4};
//     int row_ptr[] = {0, 2, 3, 3};

//     double x[] = {1.0, 2.0, 3.0};
//     double y[3];
//     printf("Matrix in dense format:\n");
//     print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, row_ptr);

//     printf("Vector:\n");
//     print_double_vector(x, 3);
//     mv_csr_symmetric(n, diag, upper, col_index, row_ptr, x, y);

//     // Calcolo atteso:
//     // y[0] = 2*1 + 1*2 + 2*3 = 2 + 2 + 6 = 10
//     // y[1] = 3*2 + 1*1 + 3*3 = 6 + 1 + 9 = 16
//     // y[2] = 4*3 + 2*1 + 3*2 = 12 + 2 + 6 = 20

//     assert(fabs(y[0] - 10.0) < 1e-6);
//     assert(fabs(y[1] - 16.0) < 1e-6);
//     assert(fabs(y[2] - 20.0) < 1e-6);

//     printf("Test 3 (Matrix-vector product CSR symemrtic): PASSED\n");
// }

void test_coo_to_csr_with_print()
{
    // define COO 
    int n = 4; 
    int upper_count = 3;
    // double diag[] = {1, 2, 3, 4};
    // double upper[] = {5, 6, 7}; 
    // int coo_row_inx[] = {0, 0, 2};
    // int col_index[] = {2, 3, 3};

    double diag[] = {1, 2, 3, 4};
    double upper[] = {6, 5, 7}; 
    int coo_row_inx[] = {0, 0, 2};
    int col_index[] = {3, 2, 3};

    int csr_row_ptr[n + 1];
    int col_inx_ordered[3];
    double upper_ordered[3];

    // coo_to_csr(n , upper_count, upper, coo_row_inx, col_index, csr_row_ptr);
    // new_coo_to_csr(n, upper_count, upper, coo_row_inx, col_index, csr_row_ptr, col_inx_ordered, upper_ordered);
    coo_to_csr(n, upper_count, upper, coo_row_inx, col_index, csr_row_ptr);
    printf("\nCSR Matrix in dense format:\n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, col_index, csr_row_ptr);

    printf("\ndiag[]: ");
    print_double_vector(diag, n);
    printf("\nupper[]: ");
    print_double_vector(upper, upper_count);
    printf("\nupper_ordered[]: ");
    print_double_vector(upper_ordered, upper_count);
    printf("\ncoo_row_inx[]: ");
    print_integer_vector(coo_row_inx, upper_count);
    printf("\ncsr_row_ptr[]: ");
    print_integer_vector(csr_row_ptr, n);
    printf("\ncol_index[]: ");
    print_integer_vector(col_index, upper_count);
    printf("\ncol_ordered[]: ");
    print_integer_vector(col_inx_ordered, upper_count);

    printf("\nCOO Matrix in dense format:\n");
    print_dense_symmeric_matrix_from_coo(n, diag, upper, coo_row_inx, col_index, upper_count);

    double x[] = {1,2,3,4};
    double out_csr[n];
    mv_csr_symmetric(n, diag, upper, col_index, csr_row_ptr, x, out_csr);
    print_double_vector(out_csr, n);

    double out_coo[n];
    mv_coo_symmetric(n, upper_count, diag, upper, coo_row_inx, col_index, x, out_coo);
    print_double_vector(out_coo, n);

    /* 
        out[0] = 1*1 + 5*3 + 6*4 = 1 + 15 + 24 = 40
        out[1] = 2*2 = 4
        out[2] = 5*1 + 3*3 + 7*4 = 5 + 9 + 28 = 42
        out[3] = 6*1 + 7*3 + 4*4 = 6 + 21 + 16 = 43
        
    */

    // assert(fabs(out[0] == 40.0) < 1e-6);
    // assert(fabs(out[1] == 4.0) < 1e-6);
    // assert(fabs(out[2] == 42.0) < 1e-6);
    // assert(fabs(out[3] == 43.0) < 1e-6);
}

int main()
{
    // test_generate_sparse_symmetric_csr();
    // test_get_matrix_entry_symmetric_csr();
    // test_mv_csr_symmetric();

    test_coo_to_csr_with_print();

    // printf("All CSR tests completed with success!\n");
    return 0;
}
