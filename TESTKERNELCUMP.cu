/*
 ============================================================================
 Name        : TESTKERNELCUMP.cu
 Author      : amakje
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <gmp.h>
#include <cump/cump.cuh>



//static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
//#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

using cump::mpf_array_t;


__global__ void cump_scal_kernel(int n, mpf_array_t alpha, mpf_array_t x) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        mpf_mul(x[idx], alpha[0], x[idx]);
        idx += gridDim.x * blockDim.x;
    }
}






int main ()
{
	int n=256;
	int prec = 20;
	int seed =20;
	 mpf_set_default_prec(prec);
	    cumpf_set_default_prec(prec);

	    //Execution configuration
	    int threads = 64;
	    int blocks = n / threads + (n % threads ? 1 : 0);
	    gmp_randstate_t  rstate;

	    //Host data
	    mpf_t *hx = new mpf_t[n];
	    mpf_t halpha;

	    //GPU data
	    cumpf_array_t dx;
	    cumpf_array_t dalpha;

	    cumpf_array_init(dx, n);
	    cumpf_array_init(dalpha, 1);


				  gmp_randinit_default (rstate);
	      		  gmp_randseed_ui (rstate, seed);

	      		mpf_init (halpha);
			  mpf_urandomb (halpha, rstate, prec);

	      		  for (int i = 0;  i < n;  ++i)
	      		    {
	      		      mpf_init (hx [i]);
	      		      mpf_urandomb (hx [i], rstate, prec);

	      		    }
	      		gmp_randclear (rstate);
	      		gmp_printf ("HX: %.70Ff \n", hx[0]);
	      		gmp_printf ("HALPHA: %.70Ff \n", halpha);

	    //Copying alpha to the GPU
	    cumpf_array_set_mpf(dalpha, &halpha, 1);

	        cumpf_array_set_mpf(dx, hx, n);


	        cump_scal_kernel<<<blocks, threads>>>(n, dalpha, dx);
	        cudaError_t errSync  = cudaGetLastError();
	        	cudaError_t errAsync = cudaDeviceSynchronize();
	        	if (errSync != cudaSuccess)
	        	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	        	if (errAsync != cudaSuccess)
	        	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));


	    //Copying to the host
	    mpf_array_set_cumpf(hx, dx, n);
	    for(int i = 1; i < n; i ++){
	        mpf_add(hx[0], hx[i], hx[0]);
	    }
	    gmp_printf ("HX RESULT: %.70Ff \n", hx[0]);

	    //Cleanup
	    mpf_clear(halpha);
	    for(int i = 0; i < n; i ++){
	        mpf_clear(hx[i]);
	    }
	    delete [] hx;
	    cumpf_array_clear(dalpha);
	    cumpf_array_clear(dx);


	return 0;
}


