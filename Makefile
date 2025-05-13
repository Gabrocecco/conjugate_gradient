CC=gcc
CFLAGS=-std=c99 -Wall -pedantic
DEBUG_FLAGS=-O0 -g -fsanitize=address
OPT_FLAGS=-O3 -DNDEBUG -march=native -mtune=native

all: debug

debug: main.c tests.c parser.c vec.c
	$(CC) $(CFLAGS) -o main main.c tests.c parser.c vec.c -lm $(DEBUG_FLAGS)

release:
	$(CC) $(CFLAGS) -o main main.c tests.c parser.c vec.c -lm $(OPT_FLAGS)

clean:
	rm -f main

test: 
	