/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_2 400 // the larger block size taylored for L2 cache
#define BLOCK_SIZE_1 40 //     smaller           taylored for L1 cache
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

void doubleBlock(int lda, int lda_padded, double* M_input, double* M, int transpose)
{
  int rem2 = lda_padded % BLOCK_SIZE_2; // The last block in each row/col might be not be a full block
  int nDivs2 = lda_padded / BLOCK_SIZE_2 + (rem2 != 0); // The number of BIG blocks per row of the entire matrix

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

        jBlock2*BLOCK_SIZE_2*lda_padded +
        iBlock2*width2*BLOCK_SIZE_2 +

        jBlock1*BLOCK_SIZE_1*height2 +
        iBlock1*width1*BLOCK_SIZE_1 +

        iOffset1*width1 + // NOTE: We don't transpose the inner matrix because we need the same orientation for 2x2 SIMD
        jOffset1

	:

        iBlock2*BLOCK_SIZE_2*lda_padded +
        jBlock2*height2*BLOCK_SIZE_2 +

        iBlock1*BLOCK_SIZE_1*width2 +
        jBlock1*height1*BLOCK_SIZE_1 +

        iOffset1*width1 +
        jOffset1;

      M[newIndex] = M_input[i*lda + j];
    }
  }
}

// N is the stride for B to access the next row
// K is the stride for A to access the next row
void do_2x2 (int lda, int N, int K, double* A, double* B, double* C)
{
  __m128d c1 = _mm_loadu_pd(C); // load C00_C01
  __m128d c2 = _mm_loadu_pd(C + lda); // load C10_C11

  __m128d a1 = _mm_load1_pd(A); // load A00_A00
  __m128d a2 = _mm_load1_pd(A + K); // load A10_A10

  __m128d b = _mm_loadu_pd(B); // load B00_B01

  c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b)); // accumulate C00_C01 += A00_A00*B00_B01
  c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b)); // accumulate C10_C11 += A10_A10*B00_B01

  ////////////////////////////////////////////////////////////////////////////////

  __m128d aa1 = _mm_load1_pd(A + 1); // load A01_A01
  __m128d aa2 = _mm_load1_pd(A + 1 + K); // load A11_A11

  __m128d bb = _mm_loadu_pd(B + N); // load B10_B11

  c1 = _mm_add_pd(c1, _mm_mul_pd(aa1, bb)); // accumulate C00_C01 += A01_A01*B10_B11
  c2 = _mm_add_pd(c2, _mm_mul_pd(aa2, bb)); // accumulate C10_C11 += A11_A11*B10_B11

  _mm_storeu_pd(C, c1);
  _mm_storeu_pd(C + lda, c2);
}

void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      double cij = C[i*lda + j];

      for (int k = 0; k < K; ++k)
      {
        //cij += A[i*K + k] * B[j*K + k];
        cij += A[i*K + k] * B[k*N + j];
      }

      C[i*lda + j] = cij;
    }
  }
}

void do_block_SIMD(int lda, int M, int N, int K, double* A, double *B, double* C)
{
  for (int i = 0; i < M; i += 2)
  {
    for (int j = 0; j < N; j += 2)
    {
      double* C_2x2 = C + i*lda + j;

      for (int k = 0; k < K; k += 2)
      {
	do_2x2(lda, N, K, A + i*K + k, B + k*N + j, C_2x2);
      }
    }
  }
}


void rect_dgemm_1(int lda, int M, int N, int K, double* A, double* B, double* C) // SINGLE level square_dgemm without any rearrangement of the arguments
{
  for (int i = 0; i < M; i += BLOCK_SIZE_1)
  {
    for (int j = 0; j < N; j += BLOCK_SIZE_1)
    {
      for (int k = 0; k < K; k += BLOCK_SIZE_1)
      {
        int MM = min(BLOCK_SIZE_1, M - i);
        int NN = min(BLOCK_SIZE_1, N - j);
        int KK = min(BLOCK_SIZE_1, K - k);

	do_block_SIMD
	(
	  lda, MM, NN, KK,
	  A + i*K + k*MM,
	  B + j*K + k*NN,
	  C + i*lda + j
	);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A_input, double* B_input, double* C_input)
{
  int lda_padded = lda + (lda % 2);

  double *A, *B, *C;
  posix_memalign((void**)&A, 2*sizeof(double), lda_padded*lda_padded*sizeof(double));
  posix_memalign((void**)&B, 2*sizeof(double), lda_padded*lda_padded*sizeof(double));
  posix_memalign((void**)&C, 2*sizeof(double), lda_padded*lda_padded*sizeof(double));
  doubleBlock(lda, lda_padded, A_input, A, 0);
  doubleBlock(lda, lda_padded, B_input, B, 1);

  for (int i = 0; i < lda_padded; i += BLOCK_SIZE_2)
  {
    for (int j = 0; j < lda_padded; j += BLOCK_SIZE_2)
    {
      for (int k = 0; k < lda_padded; k += BLOCK_SIZE_2)
      {
        int M = min(BLOCK_SIZE_2, lda_padded - i);
        int N = min(BLOCK_SIZE_2, lda_padded - j);
        int K = min(BLOCK_SIZE_2, lda_padded - k);

        rect_dgemm_1
	(
	  lda, M, N, K,
	  A + i*lda_padded + k*M,  
	  B + j*lda_padded + k*N,
	  C + i*lda + j
	);
      }
    }
  }

  memcpy(C_input, C, lda*lda*sizeof(double));

  free(A);
  free(B);
  free(C);
}
