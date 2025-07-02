#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <riscv_vector.h>

#define N (1 << 20) // 1M elements

static inline uint64_t read_rdcycle()
{
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

void dvec_fma(double *a, double *b, double *c, int n)
{

    size_t vlmax = vsetvlmax_e64m1();
    size_t i = 0;

    // body
    for (; i + vlmax <= N; i += vlmax)
    {
        vfloat64m1_t va = vle64_v_f64m1(&a[i], vlmax);
        vfloat64m1_t vb = vle64_v_f64m1(&b[i], vlmax);
        vfloat64m1_t vc = vle64_v_f64m1(&c[i], vlmax);
        vc = vfmacc_vv_f64m1(vc, va, vb, vlmax);
        vse64_v_f64m1(&c[i], vc, vlmax);
    }

    // tail
    if (i < N)
    {
        size_t vl = vsetvl_e64m1(N - i);
        vfloat64m1_t va = vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = vle64_v_f64m1(&b[i], vl);
        vfloat64m1_t vc = vle64_v_f64m1(&c[i], vl);
        vc = vfmacc_vv_f64m1(vc, va, vb, vl);
        vse64_v_f64m1(&c[i], vc, vl);
    }
}

int main()
{
    double *a = aligned_alloc(64, N * sizeof(double));
    double *b = aligned_alloc(64, N * sizeof(double));
    double *c = aligned_alloc(64, N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.5;
    }

    printf("Running dvec_fma with N = %d\n", N);

    // Warmup
    dvec_fma(a, b, c, N);

    uint64_t start = read_rdcycle();
    dvec_fma(a, b, c, N);
    uint64_t end = read_rdcycle();

    uint64_t cycles = end - start;
    double flops = 2.0 * N;
    double flops_per_cycle = flops / (double)cycles;

    printf("Cycles: %lu\n", cycles);
    printf("FLOPs:  %.0f\n", flops);
    printf("FLOPs/cycle: %.4f\n", flops_per_cycle);

    free(a);
    free(b);
    free(c);
    return 0;
}
