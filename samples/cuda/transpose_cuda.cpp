#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fastnum/transpose.hpp"
#include "matrix_utils.hpp"

#include <stdio.h>
#include <math.h>
#include <chrono>

int run_transpose_test(int M, int N) {
    float* A = (float*)malloc(sizeof(float)*M*N);
    float* B = (float*)malloc(sizeof(float)*M*N);

    float* cu_A = nullptr;
	float* cu_B = nullptr;

	cudaMalloc((void**)&cu_A, M * N * sizeof(*cu_A));
	cudaMalloc((void**)&cu_B, N * M * sizeof(*cu_B));

    cudaMemcpy(cu_A, A, M * N * sizeof(*cu_A), cudaMemcpyHostToDevice);

    

    random_matrix(A, M, N);

    auto t0 = std::chrono::system_clock::now();
    fastnum::cuda::transpose(M, N, cu_A, N, cu_B, M);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::system_clock::now();


    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    double time0 = double(duration0.count());


    printf("float transpose cuda time = %.3f micro second\n", time0);

    cudaFree(cu_A);
	cudaFree(cu_B);

    return 0;
}


int main(int argc, char* argv[]) {
    if(argc != 3) {
        printf("args error!\n");
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    run_transpose_test(M, N);

    return 0;
}