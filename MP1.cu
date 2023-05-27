//Michael Passler 20167458
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


int main()
{
	int Devices = 0;

	cudaGetDeviceCount(&Devices);
	for (int i = 0; i < Devices; i++) {

		int cores = 0;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		int mp = prop.multiProcessorCount;
		if (prop.minor == 1) {
			cores = mp * 48;
		}
		else cores = mp * 32; 

		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Number of streaming multi-processors: %d\n", prop.multiProcessorCount);
		printf("Numbe of cores: %d\n", cores);
		printf("Warp size: %d\n", prop.warpSize);
		printf("amount of global memory: %u\n", prop.totalGlobalMem);
		printf("amount of constant memory: %d\n", prop.totalConstMem);
		printf("amount of shared memory per block: %d\n", prop.sharedMemPerBlock);
		printf("number of registers available per block: %d\n", prop.regsPerBlock);
		printf("maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("maximum size of each dimension of a block: x = %d, y = %d, z = %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("maximum size of each dimension of a grid: x = %d, y = %d, z = %d\n\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}
}

