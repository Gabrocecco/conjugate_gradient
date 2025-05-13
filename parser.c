#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 
    This is a parser for a loading a matrix data from file in coo format. 
    Matrix is divided into: 
        diag:
        upper: 
        lower (symmetric)
    Indes are stored into: 
        upperAddr: i index 
        lowerAddr: j index

        matrix
{
    diag            2000 ( -3.29086 -4.0237 -4.00375 -4.01 ...);
    upper           3890 ( 0.864978 1.21294 1.21294 0.943508, ...);
    upperAddr       3890 ( 1 10 190 2 11 191 3, ...);
    lowerAddr       3890 ( 0 0 0 1 1 1 2 2 2 3 3 3 4 4, ...);

*/

// gcc -std=c99 -Wall -pedabtic -o parser parser.c


// this function parses the file and reads the matrix data component "target"
double* parseDoubleArray(FILE *file, const char *target_word, int *out_count) {
    char word[64];
    int count = 0;
    double *values = NULL;

    while (fscanf(file, "%63s", word) == 1) {
        if (strcmp(word, target_word) == 0) {
            if (fscanf(file, "%d", &count) != 1) {
                fprintf(stderr, "Error in parsing of the number after '%s'\n", target_word);
                return NULL;
            }

            char paren;
            if (fscanf(file, " %c", &paren) != 1 || paren != '(') {
                fprintf(stderr, "'(' missing after '%s'\n", target_word);
                return NULL;
            }

            values = malloc(count * sizeof(double));
            if (!values) {
                perror("Allocation failed");
                return NULL;
            }

            for (int i = 0; i < count; i++) {
                if (fscanf(file, "%lf", &values[i]) != 1) {
                    fprintf(stderr, "Error in the reading of the %d-th values of '%s'\n", i, target_word);
                    free(values);
                    return NULL;
                }
            }

            if (out_count) *out_count = count;
            return values;
        }
    }

    return NULL;
}

int* parseIntArray(FILE *file, const char *target_word, int *out_count) {
    char word[64];
    int count = 0;
    int *values = NULL;

    while (fscanf(file, "%63s", word) == 1) {
        if (strcmp(word, target_word) == 0) {
            if (fscanf(file, "%d", &count) != 1) {
                fprintf(stderr, "Error in parsing of the number after '%s'\n", target_word);
                return NULL;
            }

            char paren;
            if (fscanf(file, " %c", &paren) != 1 || paren != '(') {
                fprintf(stderr, "'(' missing after '%s'\n", target_word);
                return NULL;
            }

            values = malloc(count * sizeof(int));
            if (!values) {
                perror("Allocation failed");
                return NULL;
            }

            for (int i = 0; i < count; i++) {
                if (fscanf(file, "%d", &values[i]) != 1) {
                    fprintf(stderr, "Error in the reading of the %d-th values of '%s'\n", i, target_word);
                    free(values);
                    return NULL;
                }
            }

            if (out_count) *out_count = count;
            return values;
        }
    }

    return NULL;
}

double* parseDoubleArraySolution(FILE *file, const char *target_word, int *out_count) {
    char line[256];
    int count = 0;
    double *values = NULL;

    // read line by line
    while (fgets(line, sizeof(line), file)) {
        // check if we found the target word
        if (strstr(line, target_word)) {


            if (!fgets(line, sizeof(line), file)) break;

            // read number of values
            if (sscanf(line, "%d", &count) != 1) {
                fprintf(stderr, "Error parsing count after '%s'\n", target_word);
                return NULL;
            }

            // waits for the '('
            do {
                if (!fgets(line, sizeof(line), file)) {
                    fprintf(stderr, "Unexpected end of file, '(' not found\n");
                    return NULL;
                }
            } while (strchr(line, '(') == NULL);

            // Alloc memory for values
            values = malloc(count * sizeof(double));
            if (!values) {
                perror("Allocation failed");
                return NULL;
            }

            // Read values line by line
            int i = 0;
            while (i < count && fgets(line, sizeof(line), file)) {
                double val;
                if (sscanf(line, "%lf", &val) != 1) {
                    fprintf(stderr, "Error parsing value number %d\n", i);
                    free(values);
                    return NULL;
                }
                values[i++] = val;
            }

            if (i != count) {
                fprintf(stderr, "Error reading values of '%s' (read %d, expected %d)\n", target_word, i, count);
                free(values);
                return NULL;
            }

            // waits for the ')'
            do {
                if (!fgets(line, sizeof(line), file)) {
                    fprintf(stderr, "Unexpected end of file, ')' not found\n");
                    free(values);
                    return NULL;
                }
            } while (strchr(line, ')') == NULL);

            if (out_count) *out_count = count;
            return values;
        }
    }

    // Se target_word non trovata
    return NULL;
}
