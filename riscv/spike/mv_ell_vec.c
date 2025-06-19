#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "common.h"

#define N 5
#define MAX_NNZ 3

void mv_ell_symmetric_full_colmajor_vector(int n,
                                           int max_nnz_row, // max number of off-diagonal nnz in rows
                                           double *diag,
                                           double *ell_values, // ELL values (size n * max_nnz_row)
                                           uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                           double *x,          // input vector
                                           double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    /* 2) Diagonal contribution (vectorized)
    for (int i = 0; i < n; ++i)
    {
        y[i] += diag[i] * x[i];
    }
    */
    size_t vl;
    for (size_t i = 0; i < n; i += vl)
    {
        int remaining = n - i; // remaining elements to process
        // vl = max(vlmax, remaining);
        vl = __riscv_vsetvl_e64m1(remaining); // set vector length to the number of elements left to process (max vlmax)

        // load diag[i] and x[i] into vector registers
        vfloat64m1_t vdiag = __riscv_vle64_v_f64m1(&diag[i], vl); // vdiag[i] = diag[i], vx[i] = x[i]
        // load x[i] into vector register
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vl); // vx[i] = x[i]
        // load y[i] into vector register
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl); // should be zero, but safe, vy[i] = y[i]

        // perform element-wise multiplication and accumulate
        vy = __riscv_vfmacc_vv_f64m1(vy, vdiag, vx, vl); // y[i] += diag[i] * x[i]

        // store result back to y[i]
        __riscv_vse64_v_f64m1(&y[i], vy, vl); // y[i] = vy
    }

    // 3) non-diagonal contributes
    // iterate all slots (ELL columns)
    // for (int j = 0; j < max_nnz_row; ++j)
    // {
    //     // iterate all elements in a slot
    //     for (int i = 0; i < n; ++i)
    //     {
    //         int offset = j * n + i; // column-major offset

    //         int col = ell_cols[offset];

    //         // do not accumulate if we are on a 0 padded element
    //         if (col == -1) // padding
    //             continue;

    //         // off-diagonal contribute
    //         y[i] += ell_values[offset] * x[col];
    //     }
    // }

    for (int slot = 0; slot < max_nnz_row; ++slot) // ierate slots (ELL columns)
    {
        for (int j = 0; j < n; j += vl) // iterate elements in a slot
        {
            int remaining = n - j; // remaining elements to process

            // vl = max(vlmax, remaining);
            vl = __riscv_vsetvl_e64m1(remaining);
            printf("vl = %zu, remaining = %d\n", vl, remaining);

            size_t base_offset = slot * n + j;

            // 1. Load vl values from ell_values[] array, no gather needed
            vfloat64m1_t vvals = __riscv_vle64_v_f64m1(&ell_values[base_offset], vl);
            print_vfloat64_vector(vvals, vl, "vvals[]");

            // 2. Load vl column indices from ell_cols[] array, no gather needed
            vuint64m1_t vcol_idx = __riscv_vle64_v_u64m1(&ell_cols[base_offset], vl);
            print_vuint64_vector(vcol_idx, vl, "vcol_idx[]");

            vuint64m1_t scaled_idx = __riscv_vmul_vx_u64m1(vcol_idx, 8, vl);

            // 3. Build mask for patting (col == -1), 1 if valid, 0 if padding
            // printf("BEfore mask construction\n");
            vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx, (uint64_t)-1, vl);
            print_mask_from_indices(&ell_cols[base_offset], vl);
            // printf("After mask construction\n");

            // 4. gather from x[col] with index vector vcol_idx
            // printf("Before gather\n");
            // print_vector(x, n, "x (input vector)");              // print input vector x
            // printf("Address of x: %p\n", (void *)x);
            // print_vuint64_vector_raw(vcol_idx, vl, "vcol_idx (indices)"); // print column indices
            // print_vuint64_vector(vcol_idx, vl, "vcol_idx[]");
            vfloat64m1_t vz = __riscv_vfmv_v_f_f64m1(0.0, vl);  // initialize to all zeros ds
            // vx = __riscv_vluxei64_v_f64m1_m(mask, x, scaled_idx, vl);
            // print_vfloat64_vector(vz, vl, "vx (before gather)"); // print before gather
            // vx = __riscv_vluxei64_v_f64m1_m(mask, x, scaled_idx, vl);
            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_mu(  //masked gather starting with all zeros 
                mask, //  mask
                vz,   //  masked-off operand
                x,    //  base pointer
                scaled_idx,  // indices,
                vl);
            print_vfloat64_vector(vx, vl, "vx[] (after gather)"); // print before gather
            // printf("After gather\n");

            // 5. Load vl contiguous elements of y, no gather needed
            // printf("Before load y\n");
            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);
            // printf("After load y\n");

            // 6. vy += vvals * vx with mask
            vy = __riscv_vfmacc_vv_f64m1_m(mask, vy, vvals, vx, vl); // masked  (skip invalid lanes (with NaN))
            // vy = __riscv_vfmacc_vv_f64m1(vy, vvals, vx, vl);    // non maksed (accumlate zeros )

            // 7. Write y
            __riscv_vse64_v_f64m1(&y[j], vy, vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor(int n,
                                    int max_nnz_row, // max number of off-diagonal nnz in rows
                                    double *diag,
                                    double *ell_values, // ELL values (size n * max_nnz_row)
                                    uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                    double *x,          // input vector
                                    double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    // 2) diagonal contributes (element wise product + accumulate)
    for (int i = 0; i < n; ++i)
    {
        y[i] += diag[i] * x[i];
    }

    // 3) non-diagonal contributes
    // iterate all slots (ELL columns)
    for (int j = 0; j < max_nnz_row; ++j)
    {
        // iterate all elements in a slot
        for (int i = 0; i < n; ++i)
        {
            int offset = j * n + i; // column-major offset

            int col = ell_cols[offset];

            // do not accumulate if we are on a 0 padded element
            if (col == -1) // padding
                continue;

            // off-diagonal contribute
            y[i] += ell_values[offset] * x[col];
        }
    }
}

int main(void)
{
    // Diagonal
    double diag[N] = {10.0, 20.0, 30.0, 40.0, 50.0};

    double ell_values[MAX_NNZ * N] = {
        /* slot0 */ 1.0, 1.0, 3.0, 2.0, 5.0,
        /* slot1 */ 2.0, 3.0, 4.0, 4.0, 2.0,
        /* slot2 */ 0.0, 0.0, 0.0, 1.0, 0.0};

    uint64_t ell_cols[MAX_NNZ * N] = {
        /* slot0 */ 1, 0, 1, 0, 1,
        /* slot1 */ 3, 2, 3, 2, 3,
        /* slot2 */ -1, -1, -1, 4, -1};

    double x[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[N];
    double y_vectorized[N];

    // Stampa matrice completa
    print_dense_matrix_from_ell(N, MAX_NNZ, diag, ell_values, ell_cols);

    // Stampa contenuto ELL
    print_ell_format(N, MAX_NNZ, ell_values, ell_cols);

    // Esegui prodotto y = A * x
    mv_ell_symmetric_full_colmajor(
        N, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y);

    printf("Result y = A * x:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("  y[%d] = %g\n", i, y[i]);
    }

    mv_ell_symmetric_full_colmajor_vector(
        N, MAX_NNZ,
        diag, ell_values, ell_cols,
        x, y_vectorized);

    printf("Result y (vectorized) = A * x:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("  y[%d] = %g\n", i, y_vectorized[i]);
    }

    // Check if results match
    int pass = 1;
    for (int i = 0; i < N; ++i)
    {
        if (!fp_eq(y[i], y_vectorized[i], 1e-6f))
        {
            printf("FAIL: y[%d] = %g != %g (vectorized)\n", i, y[i], y_vectorized[i]);
            pass = 0;
        }
    }
    if (pass)
    {
        printf("PASS: Results match!\n");
    }
    else
    {
        printf("FAIL: Results do not match!\n");
    }

    return 0;
}
