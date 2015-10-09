/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include "xmmintrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 36
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

void do_2by2 (int lda, double* A, double* B, double* C)
{
    __m128d c1 = _mm_loadu_pd(C); // load C00_C01
    __m128d c2 = _mm_loadu_pd(C + lda); // load C10_C11
  
    __m128d a1 = _mm_load1_pd(A); // load A00_A00
    __m128d a2 = _mm_load1_pd(A + lda); // load A10_A10

    __m128d b = _mm_loadu_pd(B); // load B00_B01

    c1 = _mm_add_pd(c1, _mm_mul_pd(a1, b)); // accumulate C00_C01 += A00_A00*B00_B01
    c2 = _mm_add_pd(c2, _mm_mul_pd(a2, b)); // accumulate C10_C11 += A10_A10*B00_B01

    ////////////////////////////////////////////////////////////////////////////////

    __m128d aa1 = _mm_load1_pd(A + 1); // load A01_A01
    __m128d aa2 = _mm_load1_pd(A + 1 + lda); // load A11_A11

    __m128d bb = _mm_loadu_pd(B + lda); // load B10_B11

    c1 = _mm_add_pd(c1, _mm_mul_pd(aa1, bb)); // accumulate C00_C01 += A01_A01*B10_B11
    c2 = _mm_add_pd(c2, _mm_mul_pd(aa2, bb)); // accumulate C10_C11 += A11_A11*B10_B11

    _mm_storeu_pd(C, c1);
    _mm_storeu_pd(C + lda, c2);
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int i = 0; i < M; i += 2)
  {
    for (int j = 0; j < N; j += 2)
    {
      double* CC = C + i*lda + j;
      
      for (int k = 0; k < K; k += 2)
      {
        do_2by2(lda, A + i*lda + k, B + k*lda + j, CC);
      }

      if (K % 2 == 1)
      {
        C[(i + 0)*lda + (j + 0)] += A[(i + 0)*lda + (K - 1)] * B[(K - 1)*lda + (j + 0)];
        C[(i + 0)*lda + (j + 1)] += A[(i + 0)*lda + (K - 1)] * B[(K - 1)*lda + (j + 1)];
        C[(i + 1)*lda + (j + 0)] += A[(i + 1)*lda + (K - 1)] * B[(K - 1)*lda + (j + 0)];
        C[(i + 1)*lda + (j + 1)] += A[(i + 1)*lda + (K - 1)] * B[(K - 1)*lda + (j + 1)];
      }
    }
  }
  //printf("qwer\n");
  if (M % 2 == 1) // if M is odd, we must compute the last row of C's mini-block in the usual way
  {
    //printf("asdf\n");
    int I = M - 1;
    for (int j = 0; j < N; ++j)
    {
      double c_Ij = C[I*lda + j];

      for (int k = 0; k < K; ++k)
      {
        c_Ij += A[I*lda + k] * B[k*lda + j];
      }
      C[I*lda + j] = c_Ij;
    }
  }
  if (N % 2 == 1) // if N is odd, we must compute the last column of C's mini-block in the usual way
  {
    int J = N - 1;
    for (int i = 0; i < N; ++i)
    {
      double c_iJ = C[i*lda + J];

      for (int k = 0; k < K; ++k)
      {
        c_iJ += A[i*lda + k] * B[k*lda + J];
      }
      C[i*lda + J] = c_iJ;
    }
  }
  if (M % 2 == 1 || N % 2 == 1)
  {
    int I = M - 1;
    int J = N - 1;
    double c_IJ = C[I*lda + J];

    for (int k = 0; k < K; ++k)
    {
      c_IJ += A[I*lda + k] * B[k*lda + J];
    }
    C[I*lda + J] = c_IJ;
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
}
