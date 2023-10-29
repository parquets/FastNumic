#include "gemm.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <chrono>
#include "matrix_utils.hpp"
#include "gemm_naive.hpp"


void run_gemm_test(int M, int N, int K) {

    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * K * N);
    float* C0 = (float*)malloc(sizeof(float) * M * N);
    float* C1 = (float*)malloc(sizeof(float) * M * N);

    random_matrix(A, M, K);
    random_matrix(B, K, N);

    // for(int i=0; i<M*K;++i) A[i] = 0.3;
    // for(int i=0; i<K*N;++i) B[i] = 0.5;

    memset(C0, 0, sizeof(float) * M * N);
    memset(C1, 0, sizeof(float) * M * N);

    
    auto t0 = std::chrono::system_clock::now(); 
    fastnum::cpu::sgemm_nn(M, N, K, 1.0f, A, K, B, N, 0, C0, N);
    auto t1 = std::chrono::system_clock::now();
    sgemm_naive_nn(M, N, K, 1.0f, A, K, B, N, 0, C1, N);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count())/1000000.0;
    double time1 = double(duration1.count())/1000000.0;
    double GFLOPS0 = (((2.0*M*N/1000)*K)/1000000) / time0;
    double GFLOPS1 = (((2.0*M*N/1000)*K)/1000000) / time1;
    printf("For input M:%d, N:%d, K=%d, time0:%f s, GFLOPS0:%f\n", M, N, K,time0, GFLOPS0);
    printf("For input M:%d, N:%d, K=%d, time1:%f s, GFLOPS1:%f\n", M, N, K,time1, GFLOPS1);

    if (check_matrix(C0, C1, M, N))
        printf("Pass! Result Correctly!\n");
    else
        printf("Fail! Result is Not Correctly!\n");
}

int main(int argc, char* argv[]) {

    if(argc < 4) {
        printf("the input args is not correct!\n");
        return 0;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    run_gemm_test(M, N, K);
    return 0;
}

/*
./gemm_test 256 256 256
./gemm_test 1024 1024 1024
*/