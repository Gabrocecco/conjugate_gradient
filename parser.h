#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>

double *parseDoubleArray(FILE *file, const char *target_word, int *out_count);
int *parseIntArray(FILE *file, const char *target_word, int *out_count);

#endif // PARSER_H