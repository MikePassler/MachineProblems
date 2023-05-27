
//Michael Passler 20167458
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#define TILE_WIDTH 2

__global__ void matrixMulTile(float *A, float *B, float *C, int matrixSize) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < matrixSize && col < matrixSize) {
		float sum = 0.0f;

		for (int tile = 0; tile < matrixSize; tile += TILE_WIDTH) {
			__shared__ float As[TILE_WIDTH][TILE_WIDTH];
			__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

			if (row < matrixSize && tile + threadIdx.x < matrixSize) {
				As[threadIdx.y][threadIdx.x] = A[row * matrixSize + tile + threadIdx.x];
			}
			else {
				As[threadIdx.y][threadIdx.x] = 0.0f;
			}

			if (tile + threadIdx.y < matrixSize && col < matrixSize) {
				Bs[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * matrixSize + col];
			}
			else {
				Bs[threadIdx.y][threadIdx.x] = 0.0f;
			}

			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; k++) {
				sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
			}

			__syncthreads();
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
	printf("TILE_WIDTH: %d\n", TILE_WIDTH);
	float *h_A;
	float *h_B;
	float *h_C;

	int matrixDim = 2000;
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
	int x = matrixDim / TILE_WIDTH;
	if (matrixDim % TILE_WIDTH) {
		x = x + 1;
	}
	dim3 numBlocks(x, x);
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	// Initialize the CUDA device
	cudaSetDevice(0);
	float milliseconds = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	cudaEventRecord(start);

	matrixMulTile << <numBlocks, threadsPerBlock >> >(d_A, d_B, d_C, matrixDim);

	cudaDeviceSynchronize();
	// Record the stop event
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time in milliseconds

	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	printf("Time taken by GPU: %.2f ms\n", milliseconds);
	cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	float *CPU;
	CPU = (float*)malloc(bytes);
	int flag = 0;
	float begin, finish;

	begin = omp_get_wtime();

	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			float sum = 0;
			for (int k = 0; k < matrixDim; k++) {
				sum += h_A[i * matrixDim + k] * h_B[k * matrixDim + j];
			}
			CPU[i * matrixDim + j] = sum;
			if (CPU[i*matrixDim + j] != h_C[i*matrixDim + j]) {
				flag = 1;
			}
		}
	}

	if (flag == 1) { printf("TEST FAILED\n"); }
	else printf("TEST PASSED\n");

	finish = omp_get_wtime();
	float deltaT = (finish - begin);
	printf("CPU time: %f\n", (deltaT)* 1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

return 0;
}
