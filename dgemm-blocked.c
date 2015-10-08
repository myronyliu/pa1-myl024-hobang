/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <xmmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_2 37 // the larger block size taylored for L2 cache
#define BLOCK_SIZE_1 8 //     smaller           taylored for L1 cache
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

void print_matrix(int lda, double* matrix)
{
  for (int i = 0; i < lda; ++i)
  {
    for (int j = 0; j < lda; ++j)
    {
      printf("%f, ", matrix[i*lda + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void doubleBlock(int lda, double* M_input, double* M, int transpose)
{
  int rem2 = lda % BLOCK_SIZE_2; // The last block in each row/col might be not be a full block
  int nDivs2 = (rem2 == 0) ? lda / BLOCK_SIZE_2 : lda / BLOCK_SIZE_2 + 1; // The number of BIG blocks per row of the entire matrix

  for (int i = 0; i < lda; ++i)
  {
    // Within the entire matrix             , (i, j) is in BIG   block (iBlock2 , jBlock2 )
    // within BIG   block (iBlock2, jBlock2), (i, j) is in SMALL block (iBlock1 , jBlock1 )
    // within SMALL block (iBlock1, jBlock1), (i, j) is at position    (iOffset1, jOffset1)

    int iBlock2 = i / BLOCK_SIZE_2;
    int iOffset2 = i % BLOCK_SIZE_2;
    int height2 = (rem2 != 0 && iBlock2 == nDivs2 - 1) ? rem2 : BLOCK_SIZE_2;

    int iBlock1 = iOffset2 / BLOCK_SIZE_1;
    int iOffset1 = iOffset2 % BLOCK_SIZE_1;

    int iRem1 = height2 % BLOCK_SIZE_1;
    int iDivs1 = (iRem1 == 0) ? height2 / BLOCK_SIZE_1 : height2 / BLOCK_SIZE_1 + 1;
    int height1 = (iRem1 != 0 && iBlock1 == iDivs1 - 1) ? iRem1 : BLOCK_SIZE_1;

    for (int j = 0; j < lda; ++j)
    {
      int jBlock2 = j / BLOCK_SIZE_2;
      int jOffset2 = j % BLOCK_SIZE_2;
      int width2 = (rem2 != 0 && jBlock2 == nDivs2 - 1) ? rem2 : BLOCK_SIZE_2;

      int jBlock1 = jOffset2 / BLOCK_SIZE_1;
      int jOffset1 = jOffset2 % BLOCK_SIZE_1;

      int jRem1 = width2 % BLOCK_SIZE_1;
      int jDivs1 = (jRem1 == 0) ? width2 / BLOCK_SIZE_1 : width2 / BLOCK_SIZE_1 + 1;
      int width1 = (jRem1 != 0 && jBlock1 == jDivs1 - 1) ? jRem1 : BLOCK_SIZE_1;

      int newIndex = transpose

	?

        jBlock2*BLOCK_SIZE_2*lda +
        iBlock2*width2*BLOCK_SIZE_2 +

        jBlock1*BLOCK_SIZE_1*height2 +
        iBlock1*width1*BLOCK_SIZE_1+

        jOffset1*height1 +
        iOffset1

	:

        iBlock2*BLOCK_SIZE_2*lda +
        jBlock2*height2*BLOCK_SIZE_2 +

        iBlock1*BLOCK_SIZE_1*width2 +
        jBlock1*height1*BLOCK_SIZE_1+

        iOffset1*width1 +
        jOffset1;

      M[newIndex] = M_input[i*lda + j];
    }
  }
}


void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int i = 0; i < M; ++i) // For each row i of A
  {
    for (int j = 0; j < N; ++j) // For each column j of B
    {
      /* Compute C(i,j) */

      double cij = C[i*lda + j];

      for (int k = 0; k < K; ++k)
      {
        cij += A[i*K + k] * B[j*K + k];
      }

      C[i*lda + j] = cij;
    }
  }
}


inline void rect_dgemm_1(int lda, int M, int N, int K, double* A, double* B, double* C) // SINGLE level square_dgemm without any rearrangement of the arguments
{
  for (int i = 0; i < M; i += BLOCK_SIZE_1) // For each block-row of A
  {
    for (int j = 0; j < N; j += BLOCK_SIZE_1) // For each block-column of B
    {
      for (int k = 0; k < K; k += BLOCK_SIZE_1) // Accumulate block dgemms into block of C
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int MM = min(BLOCK_SIZE_1, M - i);
        int NN = min(BLOCK_SIZE_1, N - j);
        int KK = min(BLOCK_SIZE_1, K - k);
        /* Perform individual block dgemm */
        double* A_block_start = A + i*K + k*MM;
        double* B_block_start = B + j*K + k*NN;
        double* C_block_start = C + i*lda + j; // note that the stride for C is still the size of the entire matrix
        do_block(lda, MM, NN, KK, A_block_start, B_block_start, C_block_start);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A_input, double* B_input, double* C)
{
  double* A = (double*)malloc(lda*lda*sizeof(double));
  double* B = (double*)malloc(lda*lda*sizeof(double));
  doubleBlock(lda, A_input, A, 0);
  doubleBlock(lda, B_input, B, 1);
  //print_matrix(lda, A_input);
  //print_matrix(lda, A);
  //print_matrix(lda, B_input);
  //print_matrix(lda, B);

  for (int i = 0; i < lda; i += BLOCK_SIZE_2) // For each block-row of A
  {
    for (int j = 0; j < lda; j += BLOCK_SIZE_2) // For each block-column of B
    {
      for (int k = 0; k < lda; k += BLOCK_SIZE_2) // Accumulate block dgemms into block of C
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZE_2, lda - i);
        int N = min(BLOCK_SIZE_2, lda - j);
        int K = min(BLOCK_SIZE_2, lda - k);

        /* Go one level deeper */
        double* A_block_start = A + i*lda + k*M;
        double* B_block_start = B + j*lda + k*N;
        double* C_block_start = C + i*lda + j;
        
        rect_dgemm_1(lda, M, N, K, A_block_start, B_block_start, C_block_start);
      }
    }
  }

  free(A);
  free(B);
}
