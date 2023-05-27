//Michael Passler (20167458)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h> // For time()
#include <omp.h>

__global__ void matrixMul(float *A, float *B, float *C, int matrixSize) {
	// Compute the row and column indices of the element
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Perform matrix multiplication only for indices within the matrix dimensions
	if (row < matrixSize && col < matrixSize) {
		float sum = 0.0f;
		for (int k = 0; k < matrixSize; k++) {
			float a = A[row * matrixSize + k];
			float b = B[k * matrixSize + col];
			sum += a * b;
		}
		C[row * matrixSize + col] = sum;
	}

}

void populateMatrix(float *X, int matrixDim) {
	srand(time(NULL));

	// Populate the matrix with random floating point values
	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			X[i*matrixDim + j] = (float)rand() / RAND_MAX; // Generate a random floating point value between 0 and 1
		}
	}
}

int main()
{

	float *h_A;
	float *h_B;
	float *h_C;

	int matrixDim = 125;
	int matrixSize = matrixDim*matrixDim;
	int bytes = matrixSize*sizeof(float);

	//printf("bytes specified");
	//cudaGetLastError();
	cudaMallocHost((void**)&h_A, bytes);
	cudaMallocHost((void**)&h_B, bytes);
	cudaMallocHost((void**)&h_C, bytes);
	//printf("host mem allocated");

	float *d_A;
	float *d_B;
	float *d_C;

	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);
	//printf("device mem allocated");

	//fill h_A and h_B and h_C
	populateMatrix(h_A, matrixDim);
	populateMatrix(h_B, matrixDim);
	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			h_C[i*matrixDim + j] = 0;
		}
	}

	//PART 1 -------------------------------------------------------------------------------------------------
	/*
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	float milliseconds = 0.0f;
	// Record the start event
	cudaEventRecord(start);

	cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	// Initialize the CUDA device
	cudaSetDevice(0);

	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time in milliseconds

	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	printf("GPU host to device: %.2f ms\n", milliseconds);

	//cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	//now time the cpu addition and compare.
	*/
	//PART1A-------------------------------------------------------------------------------------------------------------
	/*
	cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	// Initialize the CUDA device
	cudaSetDevice(0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	float milliseconds = 0.0f;
	// Record the start event
	cudaEventRecord(start);

	cudaMemcpyAsync(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

	// Initialize the CUDA device
	cudaSetDevice(0);

	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time in milliseconds

	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	printf("GPU device to host: %.2f ms\n", milliseconds);
	*/
//PART 2----------------------------------------------------------------------------------------------------------

	cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(1, matrixSize, 1);
	dim3 numBlocks(1, 1, 1);
	// Initialize the CUDA device
	cudaSetDevice(0);
	float milliseconds = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	matrixMul << <numBlocks, threadsPerBlock >> >(d_A, d_B, d_C, matrixSize);


	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time in milliseconds

	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	printf("Time taken by GPU: %.2f ms\n", milliseconds);
	
	cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
	
	/*
	float *CPU;
	CPU = (float*)malloc(bytes);
	int flag = 0;
	float begin, finish;

	begin = omp_get_wtime();


	if (flag == 1) { printf("failed\n"); }
	else printf("succeeded\n");

	finish = omp_get_wtime();
	float deltaT = (finish - begin);
	printf("CPU time: %f\n", (deltaT)* 1000);




	*/


	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("DONE\n");

	return 0;


}
