
#include "fastnum/gemv.hpp"
#include "fastnum/gemm.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <string>
#include "gemm_naive.hpp"
#include "matrix_utils.hpp"

void run_sgemv_test(int M, int N) {
    float* A = (float*)malloc(sizeof(float) * M * N);
    float* B = (float*)malloc(sizeof(float) * N);

    float* C0 = (float*)malloc(sizeof(float) * M * 1);
    float* C1 = (float*)malloc(sizeof(float) * M * 1);

    random_matrix(A, M, N);
    random_matrix(B, N, 1);
    // for(int i=0; i<M*N;++i) A[i] = 0.3;
    // for(int i=0; i<N;++i) B[i] = 0.5;

    memset(C0, 0, sizeof(float) * M );
    memset(C1, 0, sizeof(float) * M);

    auto t0 = std::chrono::system_clock::now();
    fastnum::cpu::sgemv_n(M, N, 0.5f, A, N, B, 0, C0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::sgemm_nn(M, 1, N, 0.5f, A, N, B, 1, 0.0f, C1, 1);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;
    double GFLOPS0 = (((2.0 * M * N / 1000) * 1) / 1000000) / time0;
    double GFLOPS1 = (((2.0 * M * N / 1000) * 1) / 1000000) / time1;
    printf("For input M:%d, N:%d, time0:%f s, GFLOPS0:%f\n", M, N, time0, GFLOPS0);
    printf("For input M:%d, N:%d, time1:%f s, GFLOPS1:%f\n", M, N, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, 1))
        printf("Pass! Result Correctly!\n");
    else
        printf("Fail! Result is Not Correctly!\n");
}

void run_dgemv_test(int M, int N) {
    double* A = (double*)malloc(sizeof(double) * M * N);
    double* B = (double*)malloc(sizeof(double) * N);

    double* C0 = (double*)malloc(sizeof(double) * M * 1);
    double* C1 = (double*)malloc(sizeof(double) * M * 1);

    random_matrix(A, M, N);
    random_matrix(B, N, 1);
    // for(int i=0; i<M*N;++i) A[i] = 0.3;
    // for(int i=0; i<N;++i) B[i] = 0.5;

    memset(C0, 0, sizeof(double) * M );
    memset(C1, 0, sizeof(double) * M);

    auto t0 = std::chrono::system_clock::now();
    fastnum::cpu::dgemv_t(N, M, 0.5, A, M, B, 0, C0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::dgemm_tn(N, 1, M, 0.5, A, M, B, 1, 0.0, C1, 1);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;
    double GFLOPS0 = (((2.0 * M * N / 1000) * 1) / 1000000) / time0;
    double GFLOPS1 = (((2.0 * M * N / 1000) * 1) / 1000000) / time1;
    printf("For input M:%d, N:%d, time0:%f s, GFLOPS0:%f\n", M, N, time0, GFLOPS0);
    printf("For input M:%d, N:%d, time1:%f s, GFLOPS1:%f\n", M, N, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, 1))
        printf("Pass! Result Correctly!\n");
    else
        printf("Fail! Result is Not Correctly!\n");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("the input args is not correct!\n");
        return 0;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    // int K = atoi(argv[3]);

    run_sgemv_test(M, N);
    run_dgemv_test(M, N);
    return 0;
}

/*
.\gemv_test.exe 4096 4096
*/