#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "common.h"

/*
riscv64-unknown-elf-gcc   -O0 -march=rv64gcv -mabi=lp64d   -std=c99 -Wall -pedantic -Iinclude   -o build/test_cg_vec   src/vectorized.c   src/cg_vec.c   src/vec.c   src/coo.c   src/csr.c   src/ell.c   src/utils.c   src/parser.c   tests/test_cg_vec.c   -lm
spike --isa=rv64gcv pk build/test_cg_vec
*/

void mv_ell_symmetric_full_colmajor_vector(int n,               // A matrix dimension (n x n)
                                           int max_nnz_row,     // max number of off-diagonal nnz in rows
                                           double *diag,        // dense diangonal 
                                           double *ell_values,  // ELL values (size n * max_nnz_row)
                                           uint64_t *ell_cols,  // ELL column indices (size n * max_nnz_row)
                                           double *x,           // input vector
                                           double *y)           // output vector
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

            size_t base_offset = slot * n + j;

            // 1. Load vl values from ell_values[] array, no gather needed
            vfloat64m1_t vvals = __riscv_vle64_v_f64m1(&ell_values[base_offset], vl);

            // 2. Load vl column indices from ell_cols[] array, no gather needed
            vuint64m1_t vcol_idx = __riscv_vle64_v_u64m1(&ell_cols[base_offset], vl);

            vuint64m1_t scaled_idx = __riscv_vmul_vx_u64m1(vcol_idx, 8, vl);

            // 3. Build mask for patting (col == -1), 1 if valid, 0 if padding
            vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx, (uint64_t)-1, vl);

            // 4. gather from x[col] with index vector vcol_idx

            //! --------------------------------------------------------------------------------------
            //! This can be used with the unmasked FMA
            // vfloat64m1_t vz = __riscv_vfmv_v_f_f64m1(0.0, vl);  // initialize to all zeros ds
            // vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_mu(  //masked gather starting with all zeros
            //     mask, //  mask
            //     vz,   //  masked-off operand
            //     x,    //  base pointer
            //     scaled_idx,  // indices,
            //     vl);
            //! --------------------------------------------------------------------------------------

            //! --------------------------------------------------------------------------------------
            //! This can be used only with the masked FMA,
            //! it stores NaN's in padding positions
            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_m(mask, x, scaled_idx, vl);
            //! --------------------------------------------------------------------------------------

            // 5. Load vl contiguous elements of y, no gather needed
            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);

            // 6. vy += vvals * vx with mask
            //! --------------------------------------------------------------------------------------
            vy = __riscv_vfmacc_vv_f64m1_m(mask, vy, vvals, vx, vl); // masked  (skip invalid lanes (with NaN))
            //! --------------------------------------------------------------------------------------

            //! --------------------------------------------------------------------------------------
            // vy = __riscv_vfmacc_vv_f64m1(vy, vvals, vx, vl);    // non maksed (accumulate zeros) //! this dones't work if you have NaN's in padded positions in x
            //! --------------------------------------------------------------------------------------

            // 7. Write y
            __riscv_vse64_v_f64m1(&y[j], vy, vl);
        }
    }
}