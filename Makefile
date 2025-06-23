
# x86 compilation  
CC=gcc
CFLAGS=-std=c99 -Wall -pedantic -Iinclude	# -Iinclude searches for header filers in /include
DEBUG_FLAGS=-O0 -g -fsanitize=address,undefined 
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

# spike compilation  
RISCV_CC     := riscv64-unknown-elf-gcc
RISCV_FLAGS  := -O0 -march=rv64gcv -mabi=lp64d

RISCV_FLAGS_OPT := -O0 -march=rv64gcv -mabi=lp64d

SPIKERUN     := spike --isa=rv64gcv pk

BUILD_DIR=build

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


# === Build directory ===
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# === Clean ===
clean:
	rm -rf $(BUILD_DIR)
