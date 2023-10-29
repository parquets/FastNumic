
#include "fastnum/gemv.hpp"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <string>
#include "gemm_naive.hpp"
#include "matrix_utils.hpp"

void run_gemv_test(int M, int N) {
    float* A = (float*)malloc(sizeof(float) * M * N);
    float* B = (float*)malloc(sizeof(float) * N);

    float* C0 = (float*)malloc(sizeof(float) * M * 1);
    float* C1 = (float*)malloc(sizeof(float) * M * 1);

    random_matrix(A, M, N);
    random_matrix(B, N, 1);

    memset(C0, 0, sizeof(float) * M );
    memset(C1, 0, sizeof(float) * M);

    auto t0 = std::chrono::system_clock::now();
    fastnum::cpu::sgemv_n(M, N, 1.0, A, N, B, 0, C0);
    auto t1 = std::chrono::system_clock::now();
    sgemm_naive_nn(M, 1, N, 1.0f, A, N, B, 1, 0, C1, 1);
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

    run_gemv_test(M, N);
    return 0;
}

/*
.\gemv_test.exe 4096 4096
*/