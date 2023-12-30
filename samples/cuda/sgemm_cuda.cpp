#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fastnum/gemm.hpp"
#include "matrix_utils.hpp"

#include <stdio.h>
#include <math.h>
#include <chrono>


void run_gemm_nn_test(int M, int N, int K) {
    float* srcA = (float*)malloc(sizeof(float)*M*K);
    float* srcB = (float*)malloc(sizeof(float)*K*N);
    float* dstC0 = (float*)malloc(sizeof(float)*M*N);
    float* dstC1 = (float*)malloc(sizeof(float)*M*N);

    random_matrix(srcA, M, K);
    random_matrix(srcB, K, N);
    memset(dstC0, 0, sizeof(float)*M*N);
    memset(dstC1, 0, sizeof(float)*M*N);

    float* srcA_gpu = nullptr;
    float* srcB_gpu = nullptr;
    float* dstC_gpu = nullptr;

    cudaMalloc((void**)&srcA_gpu, sizeof(float)*M*K);
    cudaMalloc((void**)&srcB_gpu, sizeof(float)*K*N);
    cudaMalloc((void**)&dstC_gpu, sizeof(float)*M*N);

    cudaMemcpy(srcA_gpu, srcA, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(srcB_gpu, srcB, sizeof(float)*K*N, cudaMemcpyHostToDevice);

    cudaMemset(dstC_gpu, 0, sizeof(float)*M*N);

    auto t0 = std::chrono::system_clock::now();
    
    fastnum::cuda::sgemm_nn(M, N, K, 1.0, srcA_gpu, K, srcB_gpu, N, 0.0, dstC_gpu, N);
    //my_sgemm_nn<STEP, STRIDE, KSTEP, KSTRIDE> <<<grid, block>>>(M, N, K, 1.0f, srcA_gpu, K, srcB_gpu, N, 0.0f, dstC_gpu, N);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::system_clock::now();

    // fastnum::cpu::sgemm_nn(M, N, K, 1.0, srcA, K, srcB, N, 0.0, dstC0, N);

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    double time0 = double(duration0.count());

    printf("float sgemm_nn cuda time = %.3f micro second\n", time0);

    cudaMemcpy(dstC1, dstC_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    if(check_matrix(dstC0, dstC1, M, N)) {
        printf("pass!, result is correct!\n");
    }

    cudaFree(srcA_gpu);
    cudaFree(srcB_gpu);
    cudaFree(dstC_gpu);
    
} 



int main(int argc, char* argv[]) {
    if(argc != 4) {
        printf("args error!\n");
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    run_gemm_nn_test(M, N, K);

    return 0;
}

/*
./sgemm_cuda.exe 1024 1024 1024
./sgemm_cuda.exe 2048 2048 2048
./sgemm_cuda.exe 4096 4096 4096
./sgemm_cuda.exe 16384 16384 16384
*/