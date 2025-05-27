/*
 *   Matrix Market I/O example program
 *
 *   Create a small sparse, coordinate matrix and print it out
 *   in Matrix Market (v. 2.0) format to standard output.
 *
 *   (See http://math.nist.gov/MatrixMarket for details.)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "parser.h"

int main()
{
  printf("Start of program.\n");
  printf("-------------------------------------------------------------------\n");

  printf("Loading input data system from file...\n");
  const char *filename = "data/foam_128k.txt"; // linear system data input (COO) filename

  FILE *file = fopen(filename, "r");
  if (!file)
  {
    perror("Error in opening file");
    return -1;
  }
  else
  {
    printf("File %s opened successfully.\n", filename);
  }

  int count_diag = 0, count_upper = 0, count_lower = 0;

  // Parse all sections in a single file pass
  double *diag = parseDoubleArray(file, "diag", &count_diag);
  double *upper = parseDoubleArray(file, "upper", &count_upper);
  int *col_index = parseIntArray(file, "upperAddr", &count_upper); // upperAddr is used as col_index for the upper tringualar part in COO format
  int *row_index = parseIntArray(file, "lowerAddr", &count_lower);

  fclose(file);
  if (!diag || !upper || !row_index || !col_index)
  {
    fprintf(stderr, "Error parsing input data.\n");
    return -1;
  }
  printf("Parsed data successfully:\n");

  FILE *fptr, *fptr_diag;

  // Open a file in writing mode
  const char *fileOutputName = "data/matrix_128k_mm_upper.txt"; // linear system data input (COO) filename
  fptr = fopen(fileOutputName, "w");
  const char *fileOutputNameDiag = "data/matrix_128k_mm_diagonal.txt"; // linear system data input (COO) filename
  fptr_diag = fopen(fileOutputNameDiag, "w");

  MM_typecode matcode;
  int i;

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  int M = count_diag;   // Number of rows
  int N = count_diag;   // Number of columns
  int nz = count_upper; // Number of non-zero entries

  mm_write_banner(fptr, matcode);
  mm_write_mtx_crd_size(fptr, M, N, nz);

  /* NOTE: matrix market files use 1-based indices, i.e. first element
    of a vector has index 1, not 0.  */

  for (i = 0; i < nz; i++)
    fprintf(fptr, "%d %d %.17g\n", row_index[i] + 1, col_index[i] + 1, upper[i]);

  for (i = 0; i < M; i++)
    fprintf(fptr_diag, "%.17g\n", upper[i]);

  // Close the file
  fclose(fptr);
  fclose(fptr_diag);

  free(diag);
  free(upper);
  free(row_index);
  free(col_index);
  printf("Data written to file %s successfully.\n", fileOutputName);
  printf("Data written to file %s successfully.\n", fileOutputNameDiag);
  printf("End of program.\n");
  printf("-------------------------------------------------------------------\n");

  return 0;
}
