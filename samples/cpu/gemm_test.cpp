#include "gemm.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <chrono>
#include "matrix_utils.hpp"

void sgemm_naive_nn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                    float* C, int ldc) {

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

void sgemm_naive_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                    float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[m * lda + k] * B[n * ldb + k];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

void sgemm_naive_tn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                    float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[k * lda + m] * B[k * ldb + n];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

void sgemm_naive_tt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                    float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[k * lda + m] * B[n * ldb + k];
            }
            C[m * ldc + n] = tmp;
        }
    }
}

void run_gemm_test(int M, int N, int K) {

    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * K * N);
    float* C0 = (float*)malloc(sizeof(float) * M * N);
    float* C1 = (float*)malloc(sizeof(float) * M * N);

    // random_matrix(A, M, K);
    // random_matrix(B, K, N);

    for (int i = 0; i < M * K; ++i)
        A[i] = 0.3;
    for (int i = 0; i < K * N; ++i)
        B[i] = 0.5;

    memset(C0, 0, sizeof(float) * M * N);
    memset(C1, 0, sizeof(float) * M * N);

    
    auto start = std::chrono::system_clock::now(); 
    fastnum::cpu::sgemm_nn(M, N, K, 1.0f, A, K, B, N, 0, C0, N);
    //sgemm_naive_nn(M, N, K, 1.0f, A, K, B, N, 0, C1, N);
    auto end = std::chrono::system_clock::now();
    //sgemm_naive_nn(M, N, K, 1.0f, A, K, B, N, 0, C1, N);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time = double(duration.count())/1000000.0;
    double GFLOPS = (((2.0*M*N/1000)*K)/1000000) / time;
    printf("For input M:%d, N:%d, K=%d, time:%f s, GFLOPS:%f\n", M, N, K,time, GFLOPS);

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
./gemm_test 1024 512 1024
./gemm_test 1024 1024 1024
*/