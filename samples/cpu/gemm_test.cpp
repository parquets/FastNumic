#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include "gemm.hpp"

void run_gemm_test(int M, int N, int K) {

    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * K * N);
    float* C = (float*)malloc(sizeof(float) * M * N);

    fastnum::cpu::sgemm_nn(M, N, K, 1.0f, A, K, B, N, 0, C, N);
}


int main(int argc, char* argv[]) {
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    run_gemm_test(M, N, K);
    return 0;
}

/*
./gemm_test 1024 512 1024
*/