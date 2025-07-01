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

// VLMAX = 2 (128b)

// void vec_axpy(double *a, double *b, double alpha, double *out, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         out[i] = a[i] + alpha * b[i];
//     }
// }
void vec_axpy_vectorized(double *a, double *b, double alpha, double *out, int n)
{
    size_t vl;
    for (size_t i = 0; i < n; i += vl)
    {
        int remaining = n - i;
        vl = __riscv_vsetvl_e64m1(remaining);

        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);

        /*
        vfloat64m1_t __riscv_vfmacc_vf_f64m1(
            vfloat64m1_t vd,
            float64_t rs1,
            vfloat64m1_t vs2,
            size_t vl)
        */
        vfloat64m1_t result = __riscv_vfmacc_vf_f64m1(va, alpha, vb, vl);

        __riscv_vse64_v_f64m1(&out[i], // destintion, where to write
                              result,  // input, what to write
                              vl);
    }
}

// Official vectorized version of SAXPY from tutorial
void saxpy_vec_tutorial(size_t n, const float a, const float *x, float *y)
{
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl)
    {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y, vl);
        __riscv_vse32_v_f32m8(y, __riscv_vfmacc_vf_f32m8(vy, a, vx, vl), vl);
    }
}

void saxpy_vec_tutorial_double(size_t n, const double a, const double *x, double *y)
{
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl)
    {
        vl = __riscv_vsetvl_e64m1(n);
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(x, vl);
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(y, vl);
        __riscv_vse64_v_f64m1(y, __riscv_vfmacc_vf_f64m1(vy, a, vx, vl), vl);
    }
}

void saxpy_vec_tutorial_double_vlset_opt(size_t n, const double a, const double *x, double *y)
{
    size_t vlmax = __riscv_vsetvl_e64m1(n);  // set once to the max VL supported for the type

    size_t i = 0;
    //! BODY
    for (; i + (vlmax-1)< n; i += vlmax)
    {
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vlmax);
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vlmax);
        vfloat64m1_t vy_new = __riscv_vfmacc_vf_f64m1(vy, a, vx, vlmax);
        __riscv_vse64_v_f64m1(&y[i], vy_new, vlmax);
    }

    //! TAIL
    if (i<n)
    {
        size_t vl = __riscv_vsetvl_e64m1(n-i);
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vl);
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl);
        vfloat64m1_t vy_new = __riscv_vfmacc_vf_f64m1(vy, a, vx, vl);
        __riscv_vse64_v_f64m1(&y[i], vy_new, vl);
    }
}

void vec_axpy_vectorized_debug(double *a, double *b, double alpha, double *out, int n)
{
    size_t vl;
    for (size_t i = 0; i < n; i += vl)
    {
        int remaining = n - i;
        vl = __riscv_vsetvl_e64m1(remaining);

        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);
        print_vfloat64_vector(va, vl, "va[]");
        print_vfloat64_vector(vb, vl, "vb[]");
        /*
        vfloat64m1_t __riscv_vfmacc_vf_f64m1(
            vfloat64m1_t vd,
            float64_t rs1,
            vfloat64m1_t vs2,
            size_t vl)
        */
        vfloat64m1_t result = __riscv_vfmacc_vf_f64m1(va, alpha, vb, vl);
        print_vfloat64_vector(result, vl, "result[]");

        __riscv_vse64_v_f64m1(&out[i], // destintion, where to write
                              result,  // input, what to write
                              vl);
    }
}

// out = a^T * b
double vec_dot_vectorized(double *a,
                          double *b,
                          int n)
{
    // 1) compute vmax
    size_t vl_max = __riscv_vsetvl_e64m1(n);

    // 2) initiliaze a vector accumulato at zero
    vfloat64m1_t vacc = __riscv_vfmv_v_f_f64m1(0.0, vl_max);

    size_t vl;
    // 3) loop (main loop + tail)
    for (size_t i = 0; i < (size_t)n; i += vl)
    {
        size_t rem = n - i;
        vl = __riscv_vsetvl_e64m1(rem);

        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);

        // vacc += va * vb
        vacc = __riscv_vfmacc_vv_f64m1(vacc, va, vb, vl);
    }

    // 4) replicate the reduction in all elements of vacc
    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, vl_max);
    vacc = __riscv_vfredosum_vs_f64m1_f64m1(vacc, vzero, vl_max);

    // 5) extarct the first and save in result
    double result;
    __riscv_vse64_v_f64m1(&result, vacc, 1);

    return result;
}

double vec_dot_vectorized_debug(double *a,
                                double *b,
                                int n)
{
    // 1) compute vmax
    size_t vl_max = __riscv_vsetvl_e64m1(n);

    // 2) initiliaze a vector accumulato at zero
    vfloat64m1_t vacc = __riscv_vfmv_v_f_f64m1(0.0, vl_max);
    print_vfloat64_vector(vacc, vl_max, "vacc[] (before main loop)");

    size_t vl;
    // 3) loop (main loop + tail)
    for (size_t i = 0; i < (size_t)n; i += vl)
    {
        size_t rem = n - i;
        vl = __riscv_vsetvl_e64m1(rem);

        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        print_vfloat64_vector(va, vl, "va[]");
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);
        print_vfloat64_vector(va, vl, "vb[]");

        // vacc += va * vb
        vacc = __riscv_vfmacc_vv_f64m1(vacc, va, vb, vl);
        print_vfloat64_vector(va, vl, "vacc[]");
    }

    // 4) replicate the reduction in all elements of vacc
    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, vl_max);
    vacc = __riscv_vfredosum_vs_f64m1_f64m1(vacc, vzero, vl_max);
    print_vfloat64_vector(vacc, vl, "vacc[] (after reduction and brodcast)");

    // 5) extarct the first and save in result
    double result;
    __riscv_vse64_v_f64m1(&result, vacc, 1);

    return result;
}

void mv_ell_symmetric_full_colmajor_vector(int n,              // A matrix dimension (n x n)
                                           int max_nnz_row,    // max number of off-diagonal nnz in rows
                                           double *diag,       // dense diangonal
                                           double *ell_values, // ELL values all off-diagonal elements (size n * max_nnz_row)
                                           uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                           double *x,          // input vector
                                           double *y)          // output vector
{
    // initialize y at zero
    memset(y, 0, n * sizeof(double));

    // diagonal contributes
    size_t vl;
    for (size_t i = 0; i < n; i += vl) // iterate for all the diagonal elements, vl at the time
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
        // y[i] += diag[i] * x[i]
        // vfmacc.vv v[vy], v[vdiag], v[vx]
        vy = __riscv_vfmacc_vv_f64m1(vy,    // destination and accumulate target
                                     vdiag, // first input vector
                                     vx,    // secondo input vector
                                     vl);

        // store result back to y[i]
        // vse64.v v[vy], (&y[i])
        __riscv_vse64_v_f64m1(&y[i], vy, vl); // y[i] = vy
    }

    // off-diagonal contributes
    for (int slot = 0; slot < max_nnz_row; ++slot) // ierate slots (ELL columns) (there are max_nnz_row slots)
    {
        for (int j = 0; j < n; j += vl) // iterate elements in a slot (n elements for each slot)
        {
            int remaining = n - j; // remaining elements to process

            // take max from vlmax and remaining, takes care of tail
            // vl = max(vlmax, remaining);
            vl = __riscv_vsetvl_e64m1(remaining);

            // compute offset for accessing ell_values[] and ell_cols[],
            // each slot is n elements long and we are in the j-th element of this slot.
            size_t base_offset = slot * n + j;

            // 1. Load vl values from ell_values[] array starting from offset, no gather needed
            vfloat64m1_t vvals = __riscv_vle64_v_f64m1(&ell_values[base_offset], vl);

            // 2. Load vl column indices from ell_cols[] array starting from offset, no gather needed
            vuint64m1_t vcol_idx = __riscv_vle64_v_u64m1(&ell_cols[base_offset], vl);

            // scale by 8 indeces stored in vcol_idx, we need this because the gather wants the indeces in bytes positions
            // each double is 8 bytes, so we scale indeces by 8
            // from vcol_idx = [2, 4] -> scaled_idx = [128, 512]
            vuint64m1_t scaled_idx = __riscv_vmul_vx_u64m1(vcol_idx, 8, vl);

            // 3. Build mask for patting (col == -1), 1 if valid, 0 if padding
            // vcol_idx = [5, -1] -> mask = [1, 0]
            vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx,     // input vector
                                                        (uint64_t)-1, // where to mask
                                                        vl);

            /* 4. gather from x[col] with index vector vcol_idx
             masked gather from x[col] with index vector vcol_idx, leave NaN's in unmasked positions
                   x = [4,  2]
            vcol_idx = [2, -1]    --------gather------> vx = [4, NaN]
                mask = [1,  0]
            */
            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_m(mask,       // mask
                                                         x,          // input vector
                                                         scaled_idx, // indices
                                                         vl);

            // 5. Load vl contiguous elements of y, no gather needed
            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);

            // 6. vy += vvals * vx with masking, (masked FMA)
            // in masked position we dont do anything, we leave old y[i] values
            //       vx = [4,   2]
            //    vvals = [7,   0]   --------FMA------> vy = [10 + 4*7, 20]
            //       vy = [10, 20]                                       |
            //                                                        (masked)
            // vcol_idx = [2,  -1]
            //     mask = [1,   0]
            vy = __riscv_vfmacc_vv_f64m1_m(mask,  // mask
                                           vy,    // output vector where to accumulate
                                           vvals, // first input vector
                                           vx,    // second input vector
                                           vl);

            // 7. Write vy back to y
            __riscv_vse64_v_f64m1(&y[j], // destintion, where to write
                                  vy,    // input, what to write
                                  vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor_vector_vlset_opt(int n,              // A matrix dimension (n x n)
                                                    int max_nnz_row,    // max number of off-diagonal nnz in rows
                                                    double *diag,       // dense diangonal
                                                    double *ell_values, // ELL values all off-diagonal elements (size n * max_nnz_row)
                                                    uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                                    double *x,          // input vector
                                                    double *y)          // output vector
{
    // initialize y at zero
    memset(y, 0, n * sizeof(double));

    size_t vlmax = __riscv_vsetvlmax_e64m1(); // get maximum vector length for 64-bit doubles

    size_t vl = __riscv_vsetvl_e64m1(vlmax); // set initial vector length to vlmax
    size_t i = 0;

    // diagonal contributes
    // ! BODY
    for (i = 0; i + (vlmax - 1) < n; i += vl) // iterate for all the diagonal elements, vl at the time
    {
        // load diag[i] and x[i] into vector registers
        vfloat64m1_t vdiag = __riscv_vle64_v_f64m1(&diag[i], vl); // vdiag[i] = diag[i], vx[i] = x[i]
        // load x[i] into vector register
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vl); // vx[i] = x[i]
        // load y[i] into vector register
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl); // should be zero, but safe, vy[i] = y[i]

        // perform element-wise multiplication and accumulate
        // y[i] += diag[i] * x[i]
        // vfmacc.vv v[vy], v[vdiag], v[vx]
        vy = __riscv_vfmacc_vv_f64m1(vy,    // destination and accumulate target
                                     vdiag, // first input vector
                                     vx,    // secondo input vector
                                     vl);

        // store result back to y[i]
        // vse64.v v[vy], (&y[i])
        __riscv_vse64_v_f64m1(&y[i], vy, vl); // y[i] = vy
    }
    // ! TAIL
    int remaining = n - i;
    if (remaining > 0) // if there are remaining elements to process
    {
        vl = __riscv_vsetvl_e64m1(remaining); // set vector length to the number of elements left to process (max vlmax)

        // load diag[i] and x[i] into vector registers
        vfloat64m1_t vdiag = __riscv_vle64_v_f64m1(&diag[i], vl); // vdiag[i] = diag[i], vx[i] = x[i]
        // load x[i] into vector register
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vl); // vx[i] = x[i]
        // load y[i] into vector register
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl); // should be zero, but safe, vy[i] = y[i]

        // perform element-wise multiplication and accumulate
        // y[i] += diag[i] * x[i]
        // vfmacc.vv v[vy], v[vdiag], v[vx]
        vy = __riscv_vfmacc_vv_f64m1(vy,    // destination and accumulate target
                                     vdiag, // first input vector
                                     vx,    // secondo input vector
                                     vl);

        // store result back to y[i]
        // vse64.v v[vy], (&y[i])
        __riscv_vse64_v_f64m1(&y[i], vy, vl); // y[i] = vy
    }

    // off-diagonal contributes
    for (int slot = 0; slot < max_nnz_row; ++slot) // ierate slots (ELL columns) (there are max_nnz_row slots)
    {
        vl = __riscv_vsetvl_e64m1(vlmax); // set initial vector length to vlmax
        size_t j = 0;
        // ! BODY
        for (j = 0; j + (vlmax - 1) < n; j += vl) // iterate elements in a slot (n elements for each slot)
        {
            // compute offset for accessing ell_values[] and ell_cols[],
            // each slot is n elements long and we are in the j-th element of this slot.
            size_t base_offset = slot * n + j;

            // 1. Load vl values from ell_values[] array starting from offset, no gather needed
            vfloat64m1_t vvals = __riscv_vle64_v_f64m1(&ell_values[base_offset], vl);

            // 2. Load vl column indices from ell_cols[] array starting from offset, no gather needed
            vuint64m1_t vcol_idx = __riscv_vle64_v_u64m1(&ell_cols[base_offset], vl);

            // scale by 8 indeces stored in vcol_idx, we need this because the gather wants the indeces in bytes positions
            // each double is 8 bytes, so we scale indeces by 8
            // from vcol_idx = [2, 4] -> scaled_idx = [128, 512]
            vuint64m1_t scaled_idx = __riscv_vmul_vx_u64m1(vcol_idx, 8, vl);

            // 3. Build mask for patting (col == -1), 1 if valid, 0 if padding
            // vcol_idx = [5, -1] -> mask = [1, 0]
            vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx,     // input vector
                                                        (uint64_t)-1, // where to mask
                                                        vl);

            /* 4. gather from x[col] with index vector vcol_idx
             masked gather from x[col] with index vector vcol_idx, leave NaN's in unmasked positions
                   x = [4,  2]
            vcol_idx = [2, -1]    --------gather------> vx = [4, NaN]
                mask = [1,  0]
            */
            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_m(mask,       // mask
                                                         x,          // input vector
                                                         scaled_idx, // indices
                                                         vl);

            // 5. Load vl contiguous elements of y, no gather needed
            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);

            // 6. vy += vvals * vx with masking, (masked FMA)
            // in masked position we dont do anything, we leave old y[i] values
            //       vx = [4,   2]
            //    vvals = [7,   0]   --------FMA------> vy = [10 + 4*7, 20]
            //       vy = [10, 20]                                       |
            //                                                        (masked)
            // vcol_idx = [2,  -1]
            //     mask = [1,   0]
            vy = __riscv_vfmacc_vv_f64m1_m(mask,  // mask
                                           vy,    // output vector where to accumulate
                                           vvals, // first input vector
                                           vx,    // second input vector
                                           vl);

            // 7. Write vy back to y
            __riscv_vse64_v_f64m1(&y[j], // destintion, where to write
                                  vy,    // input, what to write
                                  vl);
        }
        // ! TAIL
        int remaining = n - j; // remaining elements to process
        if (remaining > 0)     // if there are remaining elements to process
        {
            vl = __riscv_vsetvl_e64m1(remaining); // set vector length to the number of elements left to process (max vlmax)
            size_t base_offset = slot * n + j;

            vfloat64m1_t vvals = __riscv_vle64_v_f64m1(&ell_values[base_offset], vl);
            vuint64m1_t vcol_idx = __riscv_vle64_v_u64m1(&ell_cols[base_offset], vl);

            vuint64m1_t scaled_idx = __riscv_vmul_vx_u64m1(vcol_idx, 8, vl);
            vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx,     // input vector
                                                        (uint64_t)-1, // where to mask
                                                        vl);

            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_m(mask,       // mask
                                                         x,          // input vector
                                                         scaled_idx, // indices
                                                         vl);

            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);

            vy = __riscv_vfmacc_vv_f64m1_m(mask,  // mask
                                           vy,    // output vector where to accumulate
                                           vvals, // first input vector
                                           vx,    // second input vector
                                           vl);

            __riscv_vse64_v_f64m1(&y[j], // destintion, where to write
                                  vy,    // input, what to write
                                  vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor_vector_m2(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diangonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    size_t vl;
    for (size_t i = 0; i < n; i += vl)
    {
        int remaining = n - i; // remaining elements to process
        // vl = max(vlmax, remaining);
        vl = __riscv_vsetvl_e64m2(remaining); // set vector length to the number of elements left to process (max vlmax)

        // load diag[i] and x[i] into vector registers
        vfloat64m2_t vdiag = __riscv_vle64_v_f64m2(&diag[i], vl); // vdiag[i] = diag[i], vx[i] = x[i]
        // load x[i] into vector register
        vfloat64m2_t vx = __riscv_vle64_v_f64m2(&x[i], vl); // vx[i] = x[i]
        // load y[i] into vector register
        vfloat64m2_t vy = __riscv_vle64_v_f64m2(&y[i], vl); // should be zero, but safe, vy[i] = y[i]

        // perform element-wise multiplication and accumulate
        vy = __riscv_vfmacc_vv_f64m2(vy, vdiag, vx, vl); // y[i] += diag[i] * x[i]

        // store result back to y[i]
        __riscv_vse64_v_f64m2(&y[i], vy, vl); // y[i] = vy
    }

    for (int slot = 0; slot < max_nnz_row; ++slot) // ierate slots (ELL columns) (max_nnz_row slots)
    {
        for (int j = 0; j < n; j += vl) // iterate elements in a slot (n elements for each slot)
        {
            int remaining = n - j; // remaining elements to process

            // take max from vlmax and remaining, takes care of tail
            // vl = max(vlmax, remaining);
            vl = __riscv_vsetvl_e64m2(remaining);

            // compute offset for accessing ell_values[] and ell_cols[],
            // each slot is n elements long and we are in the j-th element of this slot.
            size_t base_offset = slot * n + j;

            // 1. Load vl values from ell_values[] array starting from offset, no gather needed
            vfloat64m2_t vvals = __riscv_vle64_v_f64m2(&ell_values[base_offset], vl);

            // 2. Load vl column indices from ell_cols[] array starting from offset, no gather needed
            vuint64m2_t vcol_idx = __riscv_vle64_v_u64m2(&ell_cols[base_offset], vl);

            // scale by 8 indeces stored in vcol_idx, we need this because the gather wants the indeces in bytes positions
            // each double is 8 bytes, so we scale indeces by 8
            // from vcol_idx = [2, 4] -> scaled_idx = [128, 512]
            vuint64m2_t scaled_idx = __riscv_vmul_vx_u64m2(vcol_idx, 8, vl);

            // 3. Build mask for patting (col == -1), 1 if valid, 0 if padding
            // vcol_idx = [5, -1] -> mask = [1, 0]
            // vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(vcol_idx,       // input vector
            //                                             (uint64_t)-1,   // where to mask
            //                                             vl);
            /*  m1 -> vbool64_t

                m2 -> vbool32_t

                m4 -> vbool16_t*/
            vbool32_t mask = __riscv_vmsne_vx_u64m2_b32(vcol_idx, (uint64_t)-1, vl);

            /* 4. gather from x[col] with index vector vcol_idx
             masked gather from x[col] with index vector vcol_idx, leave NaN's in unmasked positions
                   x = [4,  2]
            vcol_idx = [2, -1]    --------gather------> vx = [4, NaN]
                mask = [1,  0]
            */
            vfloat64m2_t vx = __riscv_vluxei64_v_f64m2_m(mask,       // mask
                                                         x,          // input vector
                                                         scaled_idx, // indices
                                                         vl);

            // 5. Load vl contiguous elements of y, no gather needed
            vfloat64m2_t vy = __riscv_vle64_v_f64m2(&y[j], vl);

            // 6. vy += vvals * vx with masking, (masked FMA)
            // in masked position we dont do anything, we leave old y[i] values
            //       vx = [4,   2]
            //    vvals = [7,   0]   --------FMA------> vy = [10 + 4*7, 20]
            //       vy = [10, 20]

            // vcol_idx = [2,  -1]
            //     mask = [1,   0]
            vy = __riscv_vfmacc_vv_f64m2_m(mask,  // mask
                                           vy,    // output vector where to accumulate
                                           vvals, // first input vector
                                           vx,    // second input vector
                                           vl);

            // 7. Write vy back to y
            __riscv_vse64_v_f64m2(&y[j], vy, vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor_vector_m4(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diagonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    size_t vl;
    // 2) diagonal contribution
    for (size_t i = 0; i < (size_t)n; i += vl)
    {
        int remaining = n - i;
        vl = __riscv_vsetvl_e64m4(remaining);

        vfloat64m4_t vdiag = __riscv_vle64_v_f64m4(&diag[i], vl);
        vfloat64m4_t vx = __riscv_vle64_v_f64m4(&x[i], vl);
        vfloat64m4_t vy = __riscv_vle64_v_f64m4(&y[i], vl);

        vy = __riscv_vfmacc_vv_f64m4(vy, vdiag, vx, vl);
        __riscv_vse64_v_f64m4(&y[i], vy, vl);
    }

    // 3) off-diagonal (ELL) contribution
    for (int slot = 0; slot < max_nnz_row; ++slot)
    {
        for (size_t j = 0; j < (size_t)n; j += vl)
        {
            int remaining = n - j;
            vl = __riscv_vsetvl_e64m4(remaining);

            size_t base = slot * (size_t)n + j;

            // load values and column indices
            vfloat64m4_t vvals = __riscv_vle64_v_f64m4(&ell_values[base], vl);
            vuint64m4_t vcolidx = __riscv_vle64_v_u64m4(&ell_cols[base], vl);

            // scale to byte‐offsets
            vuint64m4_t scaled_idx = __riscv_vmul_vx_u64m4(vcolidx, 8, vl);

            // mask invalid lanes (col == -1)
            vbool16_t mask = __riscv_vmsne_vx_u64m4_b16(vcolidx,
                                                        (uint64_t)-1,
                                                        vl);

            // gather x[col] under mask
            vfloat64m4_t vx_g = __riscv_vluxei64_v_f64m4_m(mask,
                                                           x,
                                                           scaled_idx,
                                                           vl);

            // load y[j…]
            vfloat64m4_t vy_g = __riscv_vle64_v_f64m4(&y[j], vl);

            // vy += vvals * vx_g (masked)
            vy_g = __riscv_vfmacc_vv_f64m4_m(mask,
                                             vy_g,
                                             vvals,
                                             vx_g,
                                             vl);

            // store back to y[j…]
            __riscv_vse64_v_f64m4(&y[j], vy_g, vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor_vector_m8(int n,              // A matrix dimension (n x n)
                                              int max_nnz_row,    // max number of off-diagonal nnz in rows
                                              double *diag,       // dense diagonal
                                              double *ell_values, // ELL values (size n * max_nnz_row)
                                              uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                              double *x,          // input vector
                                              double *y)          // output vector
{
    // 1) initialize y at zero
    memset(y, 0, n * sizeof(double));

    size_t vl;
    // 2) diagonal contribution
    for (size_t i = 0; i < (size_t)n; i += vl)
    {
        int remaining = n - i;
        vl = __riscv_vsetvl_e64m8(remaining);

        vfloat64m8_t vdiag = __riscv_vle64_v_f64m8(&diag[i], vl);
        vfloat64m8_t vx = __riscv_vle64_v_f64m8(&x[i], vl);
        vfloat64m8_t vy = __riscv_vle64_v_f64m8(&y[i], vl);

        vy = __riscv_vfmacc_vv_f64m8(vy, vdiag, vx, vl);
        __riscv_vse64_v_f64m8(&y[i], vy, vl);
    }

    // 3) off-diagonal (ELL) contribution
    for (int slot = 0; slot < max_nnz_row; ++slot)
    {
        for (size_t j = 0; j < (size_t)n; j += vl)
        {
            int remaining = n - j;
            vl = __riscv_vsetvl_e64m8(remaining);

            size_t base = slot * (size_t)n + j;

            // load ELL values and column indices
            vfloat64m8_t vvals = __riscv_vle64_v_f64m8(&ell_values[base], vl);
            vuint64m8_t vcolidx = __riscv_vle64_v_u64m8(&ell_cols[base], vl);

            // scale to byte‐offsets
            vuint64m8_t scaled_idx = __riscv_vmul_vx_u64m8(vcolidx, 8, vl);

            // mask invalid lanes (col == -1)
            vbool8_t mask = __riscv_vmsne_vx_u64m8_b8(vcolidx,
                                                      (uint64_t)-1,
                                                      vl);

            // masked gather from x
            vfloat64m8_t vx_g = __riscv_vluxei64_v_f64m8_m(mask,
                                                           x,
                                                           scaled_idx,
                                                           vl);

            // load y[j…]
            vfloat64m8_t vy_g = __riscv_vle64_v_f64m8(&y[j], vl);

            // vy += vvals * vx_g (masked)
            vy_g = __riscv_vfmacc_vv_f64m8_m(mask,
                                             vy_g,
                                             vvals,
                                             vx_g,
                                             vl);

            // store back to y[j…]
            __riscv_vse64_v_f64m8(&y[j], vy_g, vl);
        }
    }
}

void mv_ell_symmetric_full_colmajor_vector_debug(int n,
                                                 int max_nnz_row, // max number of off-diagonal nnz in rows
                                                 double *diag,
                                                 double *ell_values, // ELL values (size n * max_nnz_row)
                                                 uint64_t *ell_cols, // ELL column indices (size n * max_nnz_row)
                                                 double *x,          // input vector
                                                 double *y)          // output vector
{
    size_t vlmax = __riscv_vsetvl_e64m1(n); // vlmax = 2
    printf("vlmax = %zu\n", vlmax);

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
            print_vector(x, n, "x (input vector)"); // print input vector x
            // printf("Address of x: %p\n", (void *)x);
            // print_vuint64_vector_raw(vcol_idx, vl, "vcol_idx (indices)"); // print column indices
            // print_vuint64_vector(vcol_idx, vl, "vcol_idx[]");
            // vfloat64m1_t vz = __riscv_vfmv_v_f_f64m1(0.0, vl);  // initialize to all zeros ds
            // vx = __riscv_vluxei64_v_f64m1_m(mask, x, scaled_idx, vl);
            // print_vfloat64_vector(vz, vl, "vx (before gather)"); // print before gather
            vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_m(mask, x, scaled_idx, vl);
            // vfloat64m1_t vx = __riscv_vluxei64_v_f64m1_mu(  //masked gather starting with all zeros
            //     mask, //  mask
            //     vz,   //  masked-off operand
            //     x,    //  base pointer
            //     scaled_idx,  // indices,
            //     vl);
            print_vfloat64_vector(vx, vl, "vx[] (after gather)"); // print before gather
            // printf("After gather\n");

            // 5. Load vl contiguous elements of y, no gather needed
            vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[j], vl);
            print_vfloat64_vector(vy, vl, "vy[] (before FMA)");

            // 6. vy += vvals * vx with mask
            vy = __riscv_vfmacc_vv_f64m1_m(mask, vy, vvals, vx, vl); // masked  (skip invalid lanes (with NaN))
            print_vfloat64_vector(vy, vl, "vy[] (after FMA)");
            // vy = __riscv_vfmacc_vv_f64m1(vy, vvals, vx, vl);    // non maksed (accumlate zeros )

            // 7. Write y
            __riscv_vse64_v_f64m1(&y[j], vy, vl);
        }
    }
}
