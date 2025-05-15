CC=gcc
CFLAGS=-std=c99 -Wall -pedantic
DEBUG_FLAGS=-O0 -g -fsanitize=address
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

all: debug

run: 
	./main

debug: main.c tests.c parser.c vec.c cg.c
	$(CC) $(CFLAGS) -o main main.c tests.c parser.c vec.c cg.c -lm $(DEBUG_FLAGS)
	./main 

release:
	$(CC) $(CFLAGS) -o main main.c tests.c parser.c vec.c cg.h -lm $(OPT_FLAGS)
	./main 
clean:
	rm -f main

test: tests.c vec.c cg.c 
	$(CC) $(CFLAGS) -o test tests.c vec.c cg.c -lm $(DEBUG_FLAGS)
	./test
	