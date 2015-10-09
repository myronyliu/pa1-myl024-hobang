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
        iBlock1*width1*BLOCK_SIZE_1 +

        iOffset1*width1 + // NOTE: We don't transpose the inner matrix because we need the same orientation for 2x2 SIMD
        jOffset1

	:

        iBlock2*BLOCK_SIZE_2*lda +
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
  //printf("oo\n");
  __m128d c1 = _mm_loadu_pd(C); // load C00_C01
  //printf("oo ");
  __m128d c2 = _mm_loadu_pd(C + lda); // load C10_C11
  //printf("la\n");

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
  for (int i = 0; i < M; ++i) // For each row i of A
  {
    for (int j = 0; j < N; ++j) // For each column j of B
    {
      /* Compute C(i,j) */

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
  //printf("yay\n");
  int M_padded = M + (M % 2);
  int N_padded = N + (N % 2);
  int K_padded = K + (K % 2);

  if (M != M_padded || N != N_padded || K != K_padded)
  {
    printf("BLARGH\n");
  }

  double *A_padded, *B_padded;
  posix_memalign((void**)&A_padded, 16, M_padded*K_padded);
  posix_memalign((void**)&B_padded, 16, K_padded*N_padded);

  for (int i = 0; i < M; ++i) // pad A to even width and even height
  {
    for (int j = 0; j < K; ++j)
    {
      A_padded[i*K_padded + j] = A[i*K + j];
    }
    for (int j = K; j < K_padded; ++j)
    {
      printf("asdf\n");
      A_padded[i*K_padded + j] = 0.0;
    }
  }
  for (int i = M; i < M_padded; ++i)
  {
    for (int j = 0; j < K; ++j)
    {
      printf("asdf\n");
      A_padded[i*K_padded + j] = 0.0;
    }
  }

  for (int i = 0; i < K; ++i) // pad B to even width and even height
  {
    for (int j = 0; j < N; ++j)
    {
      B_padded[i*N_padded + j] = B[i*N + j];
    }
    for (int j = N; j < N_padded; ++j)
    {
      printf("asdf\n");
      B_padded[i*N_padded + j] = 0.0;
    }
  }
  for (int i = K; i < K_padded; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("asdf\n");
      B_padded[i*N_padded + j] = 0.0;
    }
  }

  //double C_2x2[4];

  for (int i = 0; i < M_padded; i += 2)
  {
    for (int j = 0; j < N_padded; j += 2)
    {
      double* C_2x2 = C + i*lda + j;

      //printf("%d  %d\n", i, j);
      for (int k = 0; k < K_padded; k += 2)
      {
	do_2x2(lda, N_padded, K_padded, A_padded + i*K_padded + k, B_padded + k*N_padded + j, C_2x2);

	//printf("yay\n");
      }
    }
  }
}


inline void rect_dgemm_1(int lda, int M, int N, int K, double* A, double* B, double* C) // SINGLE level square_dgemm without any rearrangement of the arguments
{
  for (int i = 0; i < M; i += BLOCK_SIZE_1) // For each block-row of A
  {
    for (int j = 0; j < N; j += BLOCK_SIZE_1) // For each block-column of B
    {
      //printf("%d  %d\n", i, j);
      for (int k = 0; k < K; k += BLOCK_SIZE_1) // Accumulate block dgemms into block of C
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int MM = min(BLOCK_SIZE_1, M - i);
        int NN = min(BLOCK_SIZE_1, N - j);
        int KK = min(BLOCK_SIZE_1, K - k);

        /* Perform individual block dgemm */
	do_block
	(
	  lda, MM, NN, KK,
	  A + i*K + k*MM,
	  B + j*K + k*NN,
	  C + i*lda + j
	);

//	do_block
//	(
//	  lda, MM, NN, KK,
//	  A + i*K + k*MM,
//	  B + k*N + j*KK,
//	  C + i*lda + j
//	);
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
  //double* A = (double*)malloc(lda*lda*sizeof(double));
  //double* B = (double*)malloc(lda*lda*sizeof(double));
  double *A, *B;
  posix_memalign((void**)&A, 16, lda*lda*sizeof(double));
  posix_memalign((void**)&B, 16, lda*lda*sizeof(double));
  doubleBlock(lda, A_input, A, 0);
  doubleBlock(lda, B_input, B, 1);
  //print_matrix(lda, A_input);
  //print_matrix(lda, A);
  //print_matrix(lda, B_input);
  //print_matrix(lda, B);

  //double* C;
  //posix_memalign((void*)&C, 16, (2*lda*lda + 3)*sizeof(double));

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

        rect_dgemm_1
	(
	  lda, M, N, K,
	  A + i*lda + k*M,  
	  B + j*lda + k*N,
	  C + i*lda + j
	);

//	rect_dgemm_1
//	(
//	  lda, M, N, K,
//	  A + i*lda + k*M,
//	  B + k*lda + j*K,
//	  C + i*lda + j
//	);
      }
    }
  }

  free(A);
  free(B);
}
