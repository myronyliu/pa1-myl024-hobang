/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

//#define __SSE3__
#include <xmmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 37
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  __m128d aVec, bVec, cVec;
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; k+=2) 
      {
	double tmp[2];
#ifdef TRANSPOSE
	aVec = _mm_load_pd(&A[i*lda+k]);
	bVec = _mm_load_pd(&B[j*lda+k]);
	//cij += A[i*lda+k] * B[j*lda+k];
#else
	tmp[0] = B[k*lda+j];
	tmp[1] = B[(k+1)*lda+j];
	aVec = _mm_load_pd(&A[i*lda+k]);
	bVec = _mm_load_pd(&tmp[0]);
	//cij += A[i*lda+k] * B[k*lda+j];
#endif
	cVec = _mm_mul_pd(aVec, bVec); 
	_mm_store_pd(&tmp[0], cVec);
	cij += tmp[0] + tmp[1];
      }
      C[i*lda+j] = cij;
    }
  }
}

#define TRANSPOSE 1
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = 0; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
  }
#endif
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
