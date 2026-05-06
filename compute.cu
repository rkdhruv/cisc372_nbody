#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
#include "vector.h"
#include "config.h"
#include "compute.h"
}

#define BLOCK_DIM 16
#define REDUCE_THREADS 128

static double *d_mass = NULL;
static vector3 *d_accels = NULL;

static void checkCuda(const char *where) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
		exit(1);
	}
}

// computeAccelsKernel: one thread per (i, j) pair. 16x16 block tiles share
// 16 i-positions and 16 j-positions/masses, cutting global loads ~16x.
__global__ void computeAccelsKernel(vector3 *d_hPos, double *d_mass, vector3 *d_accels) {
	__shared__ double sPos_i[BLOCK_DIM][3];
	__shared__ double sPos_j[BLOCK_DIM][3];
	__shared__ double sMass_j[BLOCK_DIM];

	int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
	int j = blockIdx.x * BLOCK_DIM + threadIdx.x;

	// Cooperative load: one thread per row loads i data, one per column loads j data.
	if (threadIdx.x == 0 && i < NUMENTITIES) {
		sPos_i[threadIdx.y][0] = d_hPos[i][0];
		sPos_i[threadIdx.y][1] = d_hPos[i][1];
		sPos_i[threadIdx.y][2] = d_hPos[i][2];
	}
	if (threadIdx.y == 0 && j < NUMENTITIES) {
		sPos_j[threadIdx.x][0] = d_hPos[j][0];
		sPos_j[threadIdx.x][1] = d_hPos[j][1];
		sPos_j[threadIdx.x][2] = d_hPos[j][2];
		sMass_j[threadIdx.x] = d_mass[j];
	}
	__syncthreads();

	if (i >= NUMENTITIES || j >= NUMENTITIES) return;

	vector3 *cell = &d_accels[i * NUMENTITIES + j];
	if (i == j) {
		(*cell)[0] = 0;
		(*cell)[1] = 0;
		(*cell)[2] = 0;
	} else {
		double dx = sPos_i[threadIdx.y][0] - sPos_j[threadIdx.x][0];
		double dy = sPos_i[threadIdx.y][1] - sPos_j[threadIdx.x][1];
		double dz = sPos_i[threadIdx.y][2] - sPos_j[threadIdx.x][2];
		double mag_sq = dx * dx + dy * dy + dz * dz;
		double mag = sqrt(mag_sq);
		double accelmag = -GRAV_CONSTANT * sMass_j[threadIdx.x] / mag_sq;
		(*cell)[0] = accelmag * dx / mag;
		(*cell)[1] = accelmag * dy / mag;
		(*cell)[2] = accelmag * dz / mag;
	}
}

// sumAndUpdateKernel: one block per body. Threads in the block reduce
// the row of d_accels in parallel, then thread 0 updates hVel and hPos.
__global__ void sumAndUpdateKernel(vector3 *d_hPos, vector3 *d_hVel, vector3 *d_accels) {
	__shared__ double sx[REDUCE_THREADS];
	__shared__ double sy[REDUCE_THREADS];
	__shared__ double sz[REDUCE_THREADS];

	int i = blockIdx.x;
	int tid = threadIdx.x;

	double ax = 0.0, ay = 0.0, az = 0.0;
	for (int j = tid; j < NUMENTITIES; j += REDUCE_THREADS) {
		ax += d_accels[i * NUMENTITIES + j][0];
		ay += d_accels[i * NUMENTITIES + j][1];
		az += d_accels[i * NUMENTITIES + j][2];
	}
	sx[tid] = ax;
	sy[tid] = ay;
	sz[tid] = az;
	__syncthreads();

	for (int stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sx[tid] += sx[tid + stride];
			sy[tid] += sy[tid + stride];
			sz[tid] += sz[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_hVel[i][0] += sx[0] * INTERVAL;
		d_hVel[i][1] += sy[0] * INTERVAL;
		d_hVel[i][2] += sz[0] * INTERVAL;
		d_hPos[i][0] += d_hVel[i][0] * INTERVAL;
		d_hPos[i][1] += d_hVel[i][1] * INTERVAL;
		d_hPos[i][2] += d_hVel[i][2] * INTERVAL;
	}
}

extern "C" void initDeviceMemory() {
	cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
	cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	checkCuda("initDeviceMemory");
}

extern "C" void freeDeviceMemory() {
	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
	cudaFree(d_accels);
	cudaDeviceReset();
}

extern "C" void copyHostToDevice() {
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	checkCuda("copyHostToDevice");
}

extern "C" void copyDeviceToHost() {
	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	checkCuda("copyDeviceToHost");
}

extern "C" void compute() {
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid((NUMENTITIES + BLOCK_DIM - 1) / BLOCK_DIM,
	          (NUMENTITIES + BLOCK_DIM - 1) / BLOCK_DIM);
	computeAccelsKernel<<<grid, block>>>(d_hPos, d_mass, d_accels);
	sumAndUpdateKernel<<<NUMENTITIES, REDUCE_THREADS>>>(d_hPos, d_hVel, d_accels);
}
