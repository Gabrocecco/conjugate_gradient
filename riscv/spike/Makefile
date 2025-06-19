# === CONFIG ===
RISCV_PREFIX = riscv64-unknown-elf
CC      = $(RISCV_PREFIX)-gcc
CFLAGS  = -O0 -Wall -march=rv64gcv -mabi=lp64d
LDFLAGS =

SPIKE   = spike
PK      = pk

SRC     = $(wildcard *.c)
EXEC    = $(patsubst %.c, build/%, $(SRC))
NAMES   = $(patsubst %.c, %, $(SRC))

# === TARGETS ===
all: $(EXEC)

build/%: %.c | build
	$(CC) $(CFLAGS) $< -o $@

build:
	mkdir -p build

# === Auto-generated run rules ===
.PHONY: $(patsubst %, run-%, $(NAMES))

$(foreach name,$(NAMES), \
	run-$(name): build/$(name) ; \
		$(SPIKE) --isa=rv64gcv $(PK) build/$(name) \
)

# === Parametric run ===
# make run TARGET=reduction_vec
run:
	$(SPIKE) --isa=rv64gcv $(PK) build/$(TARGET)

# === Clean ===
clean:
	rm -rf build
