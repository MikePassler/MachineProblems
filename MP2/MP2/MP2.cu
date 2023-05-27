//Michael Passler (20167458)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h> // For time()
#include <omp.h>

__global__ void matrixAdd(float* A, float* B, float* C, int matrixDim) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < matrixDim && j < matrixDim) {
		int index = i * matrixDim + j;
		C[index] = A[index] + B[index];
	}
}

__global__ void matrixAddCol(float* A, float* B, float* C, int matrixDim) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < matrixDim) {
		for (int row = 0; row < matrixDim; row++) {
			int index = row * matrixDim + col;
			C[index] = A[index] + B[index];
		}
	}
	
}

__global__ void matrixAddRow(float* A, float *B, float* C, int matrixDim) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < matrixDim) {
		for (int col = 0; col < matrixDim; col++) {
			int index = row * matrixDim + col;
			C[index] = A[index] + B[index];
		}
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

	cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_C, h_C, bytes, cudaMemcpyHostToDevice);
	//printf("device mem transfered");

	// Initialize the CUDA device
	cudaSetDevice(0);

	// Define the thread block dimensions
	//dim3 threads(16, 16, 1);
	//dim3 threads(16, 1, 1);
	dim3 threads(1, 16, 1);



	int threadsPerBlock = threads.x * threads.y;


	int remainder = matrixDim / threadsPerBlock;
	int numBlocks = matrixDim / threadsPerBlock;

	if (remainder) {
		numBlocks = numBlocks + 1;
	}

	

	// Launch the kernel function with the specified thread block and grid dimensions
	int x = threads.x;
	int y = threads.y;
	//printf("%d", x);
	//printf("%d", y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	float milliseconds = 0.0f;
	// Record the start event
	cudaEventRecord(start);

	if (threads.x == 16 && threads.y == 16) {
		dim3 blocks(numBlocks, numBlocks, 1);
		matrixAdd << <blocks, threads >> >(d_A, d_B, d_C, matrixDim);
	}

	else if (threads.x == 16 && threads.y == 1) {
		dim3 blocks(numBlocks, 1, 1);
		matrixAddCol << <blocks, threads >> >(d_A, d_B, d_C, matrixDim);
	}

	else {
		dim3 blocks(1, numBlocks, 1);
		matrixAddRow << <blocks, threads >> >(d_A, d_B, d_C, matrixDim);
	}

	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time in milliseconds
	
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	printf("Time taken by GPU: %.2f ms\n", milliseconds);

	cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	//now time the cpu addition and compare.

	float *CPU;
	CPU = (float*)malloc(bytes);
	int flag = 0;
	float begin, finish;

	begin = omp_get_wtime();
	//FULL ADD and row add test
	
	for (int i = 0; i < matrixDim; i++) {
		for (int k = 0; k < matrixDim; k++) {
			int index = i*matrixDim + k;
			CPU[index] = h_A[index] + h_B[index];
			if (CPU[index] != h_C[index]) {
				flag = 1;
			}
		}
	}
	

	//COL ADD
	/*
	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			int index = j*matrixDim + i;
			CPU[index] = h_A[index] + h_B[index];
			if (CPU[index] != h_C[index]) {
				flag = 1;
			}
		}
	}
	*/
	if (flag == 1) { printf("failed\n"); }
	else printf("succeeded\n");

	finish = omp_get_wtime();
	float deltaT = (finish - begin);
	printf("CPU time: %f\n", (deltaT)*1000);


	




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


