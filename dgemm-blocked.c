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
  int k = 0; 
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      
      /* loop unrolling */
      if(K % 4 == 0)
      { 
        for (k = 0; k < K; k+=4)
        {
	  cij += A[i*block_size+k] * B[j*block_size+k];
          cij += A[i*block_size+k+1] * B[j*block_size+k+1];
          cij += A[i*block_size+k+2] * B[j*block_size+k+2];
          cij += A[i*block_size+k+3] * B[j*block_size+k+3];
        }
      } 
      else {
        for (k = 0; k < K; k++)
        {
	  cij += A[i*block_size+k] * B[j*block_size+k];
        }
      }

      C[i*lda+j] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A_input, double* B_input, double* C_input)
{
  
  double* A = (double*)malloc(2*lda*lda*sizeof(double));
  double* B = (double*)malloc(2*lda*lda*sizeof(double));
  
  //memcpy(A, A_input, lda*lda*sizeof(double));
  //memcpy(B, B_input, lda*lda*sizeof(double));

  int rem = lda % BLOCK_SIZE; // The last block in each row/col might be not be a full block
  int nDivs = (rem==0) ? lda/BLOCK_SIZE : lda/BLOCK_SIZE + 1; // The number of blocks per row/col
  
  for (int j = 0; j < lda; ++j) // First, block A in block-row major manner
  {
    int jBlock = j / BLOCK_SIZE; //0,0,1,1
    int jRem = j % BLOCK_SIZE; //0,1,0,1

    int blockWidth = (rem!=0 && jBlock==nDivs-1) ? rem : BLOCK_SIZE; 

    for (int i = 0; i < lda; ++i)
    {
      int iBlock = i / BLOCK_SIZE; //0,0,1,1
      int iRem = i % BLOCK_SIZE; //0,1,0,1
 
      int newIndex = iBlock*BLOCK_SIZE*lda +
		     jBlock*BLOCK_SIZE*BLOCK_SIZE +
	             iRem*blockWidth +
	             jRem;

      A[newIndex] = A_input[i*lda+j];
      //printf("%f  %f\n",A[newIndex],A_input[i*lda+j]); 
    }
  }
  //printf("A has been blocked\n");
/*
  printf("A_input------\n");
  print_matrix(lda, A_input);
  
  printf("A------------\n");
  print_matrix(lda, A);
*/
  for (int i = 0; i < lda; ++i) // Second, block A in block-row major manner
  {
    int iBlock = i / BLOCK_SIZE;
    int iRem = i % BLOCK_SIZE;

    int blockHeight = (rem!=0 && iBlock==nDivs-1) ? rem : BLOCK_SIZE;
 
    for (int j = 0; j < lda; ++j)
    {
      int jBlock = j / BLOCK_SIZE;
      int jRem = j % BLOCK_SIZE;

      int newIndex = jBlock*BLOCK_SIZE*lda +
		     iBlock*BLOCK_SIZE*BLOCK_SIZE +
	             jRem*blockHeight +
	             iRem;

      B[newIndex] = B_input[i*lda+j];
    }
  }
  //printf("B has been blocked\n");
/*
  printf("B_input------\n");
  print_matrix(lda, B_input);
  
  printf("B------------\n");
  print_matrix(lda, B);
*/
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
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

 	/* Perform individual block dgemm */
	do_block(K, M, N, K, (A + i*lda + k*BLOCK_SIZE), (B + j*lda + k*BLOCK_SIZE), (C_input + i*lda + j), lda);
	//do_block(BLOCK_SIZE, M, N, K, (A_input + i*lda + k*BLOCK_SIZE), (B_input + k*lda + j), (C_input + i*lda + j*BLOCK_SIZE));
      }
    }
  }

  free(A);
  free(B);
}
