
# x86 compilation  
CC=gcc
CFLAGS=-std=c99 -fgnu89-inline  -Wall -pedantic -Iinclude	# -Iinclude searches for header filers in /include
DEBUG_FLAGS=-O0 -g -fsanitize=address,undefined 
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

# spike compilation  
RISCV_CC     := riscv64-unknown-elf-gcc
RISCV_FLAGS  := -O0 -march=rv64gcv -mabi=lp64d

RISCV_FLAGS_OPT := -O0 -march=rv64gcv -mabi=lp64d

SPIKERUN     := spike --isa=rv64gcv pk

BUILD_DIR=build

# === Native RISC-V (Milk-V Pioneer) compilation ===
RISCV_NATIVE_MILKV_PIONEER_CC     := gcc

RISCV_NATIVE_MILKV_PIONEER_FLAGS  := -O3 -march=rv64gc_xtheadvector -mabi=lp64d   

# === Build all executables ===
all: $(BUILD_DIR) debug release test_cg test_coo test_csr test_ell convert_foam_to_mm mm_read

run: 
	./$(BUILD_DIR)/main

# === Debug build ===
debug: $(BUILD_DIR) src/main.c src/parser.c src/vec.c src/coo.c src/csr.c src/cg.c src/ell.c src/utils.c 
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/main_debug src/main.c src/parser.c src/vec.c src/coo.c src/csr.c src/ell.c src/cg.c src/utils.c -lm

# === Release build ===
release: $(BUILD_DIR) src/main.c src/parser.c src/vec.c src/coo.c src/csr.c src/ell.c src/cg.c src/utils.c 
	$(CC) $(CFLAGS) $(OPT_FLAGS) \
		-o $(BUILD_DIR)/main_release src/main.c src/parser.c src/vec.c src/coo.c src/csr.c src/ell.c src/cg.c src/utils.c  -lm

# === Test executables ===
test_cg: $(BUILD_DIR) tests/tests_cg.c src/vec.c src/coo.c src/cg.c src/csr.c src/ell.c src/utils.c src/parser.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/test_cg tests/tests_cg.c src/vec.c src/coo.c src/csr.c src/ell.c src/utils.c src/parser.c src/cg.c -lm

test_coo: $(BUILD_DIR) src/coo.c src/csr.c tests/test_coo.c src/utils.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/test_coo src/coo.c src/csr.c src/utils.c tests/test_coo.c  -lm

test_csr: $(BUILD_DIR) src/csr.c tests/test_csr.c src/utils.c src/coo.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/test_csr src/csr.c src/utils.c src/coo.c tests/test_csr.c -lm

test_ell: $(BUILD_DIR) tests/test_ell.c src/ell.c src/utils.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/test_ell tests/test_ell.c src/ell.c src/utils.c -lm

convert_foam_to_mm: $(BUILD_DIR) src/foam_to_mm.c src/parser.c src/mmio.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/foam_to_mm src/foam_to_mm.c src/parser.c src/mmio.c -lm

mm_read: $(BUILD_DIR) src/mm_read.c src/mmio.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) \
		-o $(BUILD_DIR)/mm_read src/mm_read.c src/mmio.c -lm


# === Test vectorized matrix vector product in ell format on RISC-V and run under Spike ===

test_vec: $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/test_vec \
		src/vectorized.c \
		src/ell.c \
		src/coo.c \
		src/csr.c \
		src/common.c \
		src/parser.c \
		src/mmio.c \
		tests/test_vec.c

run_test_vec: test_vec 
			$(SPIKERUN) $(BUILD_DIR)/test_vec

# === Test vectorized CG on RISC-V and run under Spike ===

test_cg_vec: $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_FLAGS) $(CFLAGS) \
		$(CFLAGS) \
		-o $(BUILD_DIR)/test_cg_vec \
		src/vectorized.c \
		src/common.c \
		src/cg_vec.c \
		src/vec.c \
		src/coo.c \
		src/csr.c \
		src/ell.c \
		src/utils.c \
		src/parser.c \
		tests/test_cg_vec.c \
		-lm

run_test_cg_vec: test_cg_vec
	spike --isa=rv64gcv pk $(BUILD_DIR)/test_cg_vec

# Milk-V Pionner test executable for vectorized CG
test_cg_vec_native: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/test_cg_vec_native \
		src/vectorized.c \
		src/common.c \
		src/cg_vec.c \
		src/vec.c \
		src/coo.c \
		src/csr.c \
		src/ell.c \
		src/utils.c \
		src/parser.c \
		tests/test_cg_vec.c \
		-lm

run_test_cg_vec_native: test_cg_vec_native
	./$(BUILD_DIR)/test_cg_vec_native


# Milk-V Pionner test executable for test_vec
test_vec_native: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/test_vec_native \
		src/vectorized.c \
		src/ell.c \
		src/coo.c \
		src/csr.c \
		src/common.c \
		src/parser.c \
		src/mmio.c \
		tests/test_vec.c

run_test_vec_native: test_vec_native
	./$(BUILD_DIR)/test_vec_native


# Benchmarks Serial vs Vector

benchmark_matvec_scalar_vs_vector: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/benchmark_matvec_scalar_vs_vector \
		src/vectorized.c \
		src/ell.c \
		src/coo.c \
		src/csr.c \
		src/common.c \
		src/parser.c \
		src/mmio.c \
		benchmarks/rvv_matrix_vector.c \
		-lrt

create_saxpy_asm_vec: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/$@ \
		src/vectorized.c \
		-lrt

create_saxpy_asm_scalar: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/$@ \
		benchmarks/rvv_matrix_vector.c \
		-lrt

run_benchmark_matvec_scalar_vs_vector: benchmark_matvec_scalar_vs_vector 
			$(BUILD_DIR)/benchmark_matvec_scalar_vs_vector


# Milk-V Pioneer test CG scalar vs vector
benchmark_cg_scalar_vs_vector_native: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/benchmark_cg_scalar_vs_vector_native \
		src/vectorized.c \
		src/common.c \
		src/cg_vec.c \
		src/cg.c \
		src/vec.c \
		src/coo.c \
		src/csr.c \
		src/ell.c \
		src/utils.c \
		src/parser.c \
		src/mmio.c \
		benchmarks/rvv_cg.c \
		-lm

run_benchmark_cg_scalar_vs_vector_native: benchmark_cg_scalar_vs_vector_native
	./$(BUILD_DIR)/benchmark_cg_scalar_vs_vector_native

# Peak FLOPS benchmark
benchmark_peak_flops: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/$@ \
		benchmarks/roofline/flops_peak.c

# STREAM benchmark
benchmark_stream: $(BUILD_DIR)
	gcc -O3 -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=30 benchmarks/roofline/stream.c  -o build/stream 



# Milk-V matmul example 
test_rvv_matmul: $(BUILD_DIR)
	$(RISCV_NATIVE_MILKV_PIONEER_CC) $(RISCV_NATIVE_MILKV_PIONEER_FLAGS) $(CFLAGS) \
		-o $(BUILD_DIR)/$@ \
		benchmarks/rvv_tutorials/rvv_matmul.c \
		src/common.c

run_test_rvv_matmul: test_rvv_matmul
	./$(BUILD_DIR)/test_rvv_matmul


# === Build directory ===
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# === Clean ===
clean:
	rm -rf $(BUILD_DIR)
