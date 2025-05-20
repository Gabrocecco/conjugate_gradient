#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "coo.h"
#include "csr.h"
#include "utils.h"

// Test 1: random COO matrix generation 
void test_generate_matrix()
{
    int n = 100;
    double density = 0.5;
    double *diag, *upper;
    int *i_idx, *j_idx;

    int upper_count = generate_sparse_symmetric_matrix_full_diag_coo(n, density, &diag, &upper, &i_idx, &j_idx);

    printf("Printing COO matrix in dense format: \n");
    print_dense_symmeric_matrix_from_coo(n, diag, upper, i_idx, j_idx, upper_count);
    printf("diag: \n"); 
    print_double_vector(diag, n);
    printf("upper: \n");
    print_double_vector(upper, upper_count);
    printf("row_idx: \n");
    print_integer_vector(i_idx, upper_count);
    printf("col idx: ");
    print_integer_vector(j_idx, upper_count);

    printf("Coversion COO to CSR: \n\n");
    int row_ptr[n+1];
    coo_to_csr(n, upper_count, upper, i_idx, j_idx, row_ptr);
    printf("CSR dense: \n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, j_idx, row_ptr);
    printf("row_ptr: ");
    print_integer_vector(row_ptr, n+1);

    assert(compare_symmetric_matrices_coo_csr(n,
        diag,
        upper,
        i_idx,
        j_idx,
        upper_count,
        diag,
        upper,
        j_idx,
        row_ptr));

    // print matrix in COO and dense format 
    printf("Matrix in dense format:\n");
    assert(diag != NULL);
    assert(upper != NULL);
    assert(i_idx != NULL);
    assert(j_idx != NULL);
    assert(upper_count >= 0);

    printf("Test 1 (Generation of COO matrix ): PASSED\n");

    free(diag);
    free(upper);
    free(i_idx);
    free(j_idx);
}

// Test 2: verify that get_matrix_entry_symmetric_coo returns correct values 
void test_matrix_entry_1()
{
    int n = 3;
    double diag[] = {10, 20, 30};
    double upper[] = {1, 2, 3};
    int i_idx[] = {0, 0, 1};
    int j_idx[] = {1, 2, 2};
    int upper_count = 3;

    printf("COO dense: \n");
    print_dense_symmeric_matrix_from_coo(n, diag, upper, i_idx, j_idx, upper_count);
    printf("Coversion COO to CSR: \n");
    int row_ptr[n+1];
    coo_to_csr(n, upper_count, upper, i_idx, j_idx, row_ptr);
    printf("CSR dense: \n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, j_idx, row_ptr);


    // Diagonale
    assert(get_matrix_entry_symmetric_coo(0, 0, n, diag, upper, i_idx, j_idx, upper_count) == 10);
    assert(get_matrix_entry_symmetric_coo(1, 1, n, diag, upper, i_idx, j_idx, upper_count) == 20);
    assert(get_matrix_entry_symmetric_coo(2, 2, n, diag, upper, i_idx, j_idx, upper_count) == 30);

    // Parte superiore
    assert(get_matrix_entry_symmetric_coo(0, 1, n, diag, upper, i_idx, j_idx, upper_count) == 1);
    assert(get_matrix_entry_symmetric_coo(0, 2, n, diag, upper, i_idx, j_idx, upper_count) == 2);
    assert(get_matrix_entry_symmetric_coo(1, 2, n, diag, upper, i_idx, j_idx, upper_count) == 3);

    // Parte inferiore (simmetrica)
    assert(get_matrix_entry_symmetric_coo(1, 0, n, diag, upper, i_idx, j_idx, upper_count) == 1);
    assert(get_matrix_entry_symmetric_coo(2, 0, n, diag, upper, i_idx, j_idx, upper_count) == 2);
    assert(get_matrix_entry_symmetric_coo(2, 1, n, diag, upper, i_idx, j_idx, upper_count) == 3);

    printf("Test 2 (get_matrix_entry_1): PASSED\n");
}

// Test 3: verify that get_matrix_entry_symmetric_coo returns correct values 
void test_matrix_entry_2()
{
    int n = 4;
    double diag[] = {10, 20, 30, 40};
    double upper[] = {5, 6};
    int i_idx[] = {0, 1};
    int j_idx[] = {2, 3};
    int upper_count = 2;

    print_dense_symmeric_matrix_from_coo(n, diag, upper, i_idx, j_idx, upper_count);

    

    // Diagonal 
    assert(get_matrix_entry_symmetric_coo(0, 0, n, diag, upper, i_idx, j_idx, upper_count) == 10);
    assert(get_matrix_entry_symmetric_coo(1, 1, n, diag, upper, i_idx, j_idx, upper_count) == 20);
    assert(get_matrix_entry_symmetric_coo(2, 2, n, diag, upper, i_idx, j_idx, upper_count) == 30);
    assert(get_matrix_entry_symmetric_coo(3, 3, n, diag, upper, i_idx, j_idx, upper_count) == 40);

    // Upper
    assert(get_matrix_entry_symmetric_coo(0, 1, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(0, 2, n, diag, upper, i_idx, j_idx, upper_count) == 5);
    assert(get_matrix_entry_symmetric_coo(0, 3, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(1, 2, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(1, 3, n, diag, upper, i_idx, j_idx, upper_count) == 6);
    assert(get_matrix_entry_symmetric_coo(2, 3, n, diag, upper, i_idx, j_idx, upper_count) == 0);

    // Lower (symmetric)
    assert(get_matrix_entry_symmetric_coo(1, 0, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(2, 0, n, diag, upper, i_idx, j_idx, upper_count) == 5);
    assert(get_matrix_entry_symmetric_coo(3, 0, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(2, 1, n, diag, upper, i_idx, j_idx, upper_count) == 0);
    assert(get_matrix_entry_symmetric_coo(3, 1, n, diag, upper, i_idx, j_idx, upper_count) == 6);
    assert(get_matrix_entry_symmetric_coo(3, 2, n, diag, upper, i_idx, j_idx, upper_count) == 0);

    printf("Test 3 (get_matrix_entry_2): PASSED\n");
}

// Test 4: verifica prodotto matrice-vettore
void test_mv_product()
{
    int n = 3;
    double diag[] = {10, 20, 30};
    double upper[] = {1, 2, 3};
    int i_idx[] = {0, 0, 1};
    int j_idx[] = {1, 2, 2};
    int upper_count = 3;
    print_dense_symmeric_matrix_from_coo(n, diag, upper, i_idx, j_idx, upper_count);

    double v[] = {1, 2, 3};
    double out_coo[3];
    mv_coo_symmetric(n, upper_count, diag, upper, i_idx, j_idx, v, out_coo);

    // Calcolo atteso:
    // y[0] = 10*1 + 1*2 + 2*3 = 10 + 2 + 6 = 18
    // y[1] = 20*2 + 1*1 + 3*3 = 40 + 1 + 9 = 50
    // y[2] = 30*3 + 2*1 + 3*2 = 90 + 2 + 6 = 98

    assert(out_coo[0] == 18);
    assert(out_coo[1] == 50);
    assert(out_coo[2] == 98);


    printf("Coversion COO to CSR: \n");
    int row_ptr[n+1];
    coo_to_csr(n, upper_count, upper, i_idx, j_idx, row_ptr);
    printf("CSR dense: \n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, j_idx, row_ptr);

    double out_csr[3];
    mv_csr_symmetric(n, diag, upper, j_idx, row_ptr, v, out_csr);

    assert(out_csr[0] == 18);
    assert(out_csr[1] == 50);
    assert(out_csr[2] == 98);

    printf("Test 4 (mv_coo_symmetric): PASSED\n");
}

void test_mv_random_product(){
    int n = 10;
    double density = 0.5;
    double *diag, *upper;
    int *i_idx, *j_idx;

    int upper_count = generate_sparse_symmetric_matrix_full_diag_coo(n, density, &diag, &upper, &i_idx, &j_idx);

    printf("Printing COO matrix in dense format: \n");
    print_dense_symmeric_matrix_from_coo(n, diag, upper, i_idx, j_idx, upper_count);
    // printf("diag: \n"); 
    // print_double_vector(diag, n);
    // printf("upper: \n");
    // print_double_vector(upper, upper_count);
    // printf("row_idx: \n");
    // print_integer_vector(i_idx, upper_count);
    // printf("col idx: ");
    // print_integer_vector(j_idx, upper_count);

    double v[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    printf("vector: \n");
    print_double_vector(v, 5);
    double out_coo[n];

    mv_coo_symmetric(n, upper_count, diag, upper, i_idx, j_idx, v, out_coo);
    print_double_vector(out_coo, n);


    printf("Coversion COO to CSR: \n\n");
    int row_ptr[n+1];
    coo_to_csr(n, upper_count, upper, i_idx, j_idx, row_ptr);
    printf("CSR dense: \n");
    print_dense_symmetric_matrix_from_csr(n, diag, upper, j_idx, row_ptr);
    printf("row_ptr: ");
    print_integer_vector(row_ptr, n+1);

    double out_csr[n];
    mv_csr_symmetric(n, diag, upper, j_idx, row_ptr, v, out_csr);
    print_double_vector(out_csr, n);

    compare_double_vectors_fixed_tol(out_coo, out_csr, n);

    // print matrix in COO and dense format 
    printf("Matrix in dense format:\n");
    assert(diag != NULL);
    assert(upper != NULL);
    assert(i_idx != NULL);
    assert(j_idx != NULL);
    assert(upper_count >= 0);

    printf("Test 1 (Generation of COO matrix ): PASSED\n");

    free(diag);
    free(upper);
    free(i_idx);
    free(j_idx);
}

int main()
{
    test_generate_matrix();
    // test_matrix_entry_1();
    // test_matrix_entry_2();
    // test_mv_product();
    // test_mv_random_product();

    printf("Tutti i test COO completati con successo!\n");
    return 0;
}
