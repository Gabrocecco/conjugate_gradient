CC=gcc
CFLAGS=-std=c99 -Wall -pedantic
DEBUG_FLAGS=-O0 -g -fsanitize=address
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

all: debug

run: 
	./main

run_test_matrix:
	./test_matrix

debug: main.c tests.c parser.c vec.c cg.c
	$(CC) $(CFLAGS) -o main main.c parser.c vec.c cg.c -lm $(DEBUG_FLAGS)

release:
	$(CC) $(CFLAGS) -o main main.c tests.c parser.c vec.c cg.h -lm $(OPT_FLAGS)
	 
clean:
	rm -f main

test_cg: tests.c vec.c cg.c 
	$(CC) $(CFLAGS) -o test tests.c vec.c cg.c -lm $(DEBUG_FLAGS)

test_matrix: matrix.c csr.c 
	$(CC) $(CFLAGS) -o test_matrix matrix.c csr.c -lm $(DEBUG_FLAGS)
