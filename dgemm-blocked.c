/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

//#include <x86intrin.h>
#include <pmmintrin.h> //should compile with "make OPTIMIZATION=-O3\ -msse3"
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

void print_matrix(int lda, double* matrix) {
  for(int i = 0; i < lda ; ++i) {
    for(int j = 0; j < lda ; ++j) {
      printf("%f, ",matrix[i*lda+j]); 
    }
     printf("\n");
  }  	
}

static void do_block (int block_size, int M, int N, int K, double* A, double* B, double* C, int lda)
{
   /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      __m128d c = _mm_setzero_pd();
      /* loop unrolling */
      if(K % 2 == 0)
      { 
        double temp;
        for (int k = 0; k < K; k+=2)
        {
          c = _mm_add_pd(c, _mm_mul_pd(_mm_load_pd(&A[i*block_size+k]), _mm_load_pd(&B[j*block_size+k]))); 
        } 
   
        c = _mm_hadd_pd(c, c);          
        _mm_store_pd(&temp, c);

        cij += temp;
      }
      else 
      {
        for (int k = 0; k < K; k++)
        {
	  cij += A[i*block_size+k] * B[j*block_size+k];
        }
      }

      C[i*lda+j] = cij;
    }//end for
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A_input, double* B_input, double* C_input)
{
  
  double* A = (double*)malloc(1.5*lda*lda*sizeof(double));
  double* B = (double*)malloc(1.5*lda*lda*sizeof(double));
  int iBlock, iRem, AnewIndex, BnewIndex, blockHeight, blockWidth, jBlock, jRem;  
  int M, N, K;

  int rem = lda % BLOCK_SIZE; // The last block in each row/col might be not be a full block
  int nDivs = (rem==0) ? lda/BLOCK_SIZE : lda/BLOCK_SIZE + 1; // The number of blocks per row/col
  int BLOCK_SQUARE = BLOCK_SIZE * BLOCK_SIZE;
  int BLOCK_BY_LDA = BLOCK_SIZE*lda;

  for (int j = 0; j < lda; ++j) // First, block A in block-row major manner
  {
    jBlock = j / BLOCK_SIZE; //0,0,1,1
    jRem = j % BLOCK_SIZE; //0,1,0,1

    blockWidth = (rem!=0 && jBlock==nDivs-1) ? rem : BLOCK_SIZE; 

    for (int i = 0; i < lda; ++i)
    {
      iBlock = i / BLOCK_SIZE; //0,0,1,1
      iRem = i % BLOCK_SIZE; //0,1,0,1
 
      AnewIndex = iBlock*BLOCK_BY_LDA +
		  jBlock*BLOCK_SQUARE +
	          iRem*blockWidth +
	          jRem;
      
      A[AnewIndex] = A_input[i*lda+j];
    }
  }

  for (int i = 0; i < lda; ++i) // Second, block B in block-column major manner
  {
    iBlock = i / BLOCK_SIZE;
    iRem = i % BLOCK_SIZE;

    blockHeight = (rem!=0 && iBlock==nDivs-1) ? rem : BLOCK_SIZE;
 
    for (int j = 0; j < lda; ++j)
    {
      jBlock = j / BLOCK_SIZE;
      jRem = j % BLOCK_SIZE;

      BnewIndex = jBlock*BLOCK_BY_LDA +
		  iBlock*BLOCK_SQUARE +
	          jRem*blockHeight +
	          iRem;

      B[BnewIndex] = B_input[i*lda+j];
    }
  }

  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
  {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	M = min (BLOCK_SIZE, lda-i);
	N = min (BLOCK_SIZE, lda-j);
	K = min (BLOCK_SIZE, lda-k);

 	/* Perform individual block dgemm */
	do_block(K, M, N, K, (A + i*lda + k*BLOCK_SIZE), (B + j*lda + k*BLOCK_SIZE), (C_input + i*lda + j), lda);	
      }
    }
  }

  free(A);
  free(B);
}
