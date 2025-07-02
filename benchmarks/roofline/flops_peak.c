#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#define N (1 << 20) // 1M elementi

static inline uint64_t read_rdcycle() {
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

void dvec_fma(double *a, double *b, double *c, int n) {
    size_t vlmax = __riscv_vsetvlmax_e64m1(); // massimo VL
    size_t vl = __riscv_vsetvl_e64m1(vlmax);  // set iniziale
    size_t i = 0;

    // corpo principale: processa in blocchi da vlmax
    for (; i + (vlmax - 1) < n; i += vl) {
        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);
        vfloat64m1_t vc = __riscv_vle64_v_f64m1(&c[i], vl);
        vc = __riscv_vfmacc_vv_f64m1(vc, va, vb, vl);
        __riscv_vse64_v_f64m1(&c[i], vc, vl);
    }

    // tail finale (se n non multiplo di vlmax)
    int remaining = n - i;
    if (remaining > 0) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);
        vfloat64m1_t vc = __riscv_vle64_v_f64m1(&c[i], vl);
        vc = __riscv_vfmacc_vv_f64m1(vc, va, vb, vl);
        __riscv_vse64_v_f64m1(&c[i], vc, vl);
    }
}

void check_alignment(const char *name, void *ptr) {
    if ((uintptr_t)ptr % 64 != 0)
        printf("Warning: pointer '%s' not 64-byte aligned! RVV performance may degrade.\n", name);
}

int main() {
	double *a, *b, *c;
	if (posix_memalign((void**)&a, 64, N * sizeof(double)) != 0 ||
    		posix_memalign((void**)&b, 64, N * sizeof(double)) != 0 ||
    		posix_memalign((void**)&c, 64, N * sizeof(double)) != 0) {
    		perror("posix_memalign failed");
    		exit(EXIT_FAILURE);
	}

    for (int i = 0; i < N; ++i) {
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

    free(a); free(b); free(c);
    return 0;
}
