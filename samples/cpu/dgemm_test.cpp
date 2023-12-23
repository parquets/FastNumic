#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <string>
#include "gemm.hpp"
#include "gemm_naive.hpp"
#include "matrix_utils.hpp"

void run_dgemm_test(int M, int N, int K) {
    printf("run_dgemm_test\n");
    double* A = (double*)malloc(sizeof(double) * M * K);
    double* B = (double*)malloc(sizeof(double) * K * N);
    double* C0 = (double*)malloc(sizeof(double) * M * N);
    double* C1 = (double*)malloc(sizeof(double) * M * N);

    auto t0 = std::chrono::system_clock::now();
    auto t1 = std::chrono::system_clock::now();
    auto t2 = std::chrono::system_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = 0.0;
    double time1 = 0.0;
    double GFLOPS0 = 0.0;
    double GFLOPS1 = 0.0;

    random_matrix(A, M, K);
    random_matrix(B, K, N);

    memset(C0, 0, sizeof(double) * M * N);
    memset(C1, 0, sizeof(double) * M * N);

    t0 = std::chrono::system_clock::now();
    fastnum::cpu::dgemm_nn(M, N, K, 0.5, A, K, B, N, 0, C0, N);
    t1 = std::chrono::system_clock::now();
    dgemm_naive_nn(M, N, K, 0.5, A, K, B, N, 0, C1, N);
    t2 = std::chrono::system_clock::now();

    duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    time0 = double(duration0.count()) / 1000000.0;
    time1 = double(duration1.count()) / 1000000.0;
    GFLOPS0 = (((2.0 * M * N / 1000) * K) / 1000000) / time0;
    GFLOPS1 = (((2.0 * M * N / 1000) * K) / 1000000) / time1;
    printf("dgemm_nn For input M:%d, N:%d, K=%d, time0:%f s, GFLOPS0:%f\n", M, N, K, time0, GFLOPS0);
    printf("dgemm_nn For input M:%d, N:%d, K=%d, time1:%f s, GFLOPS1:%f\n", M, N, K, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, N)) {
        printf("dgemm_nn Pass! Result Correctly!\n");
    } else {
        printf("dgemm_nn Fail! Result is Not Correctly!\n");
    }

    memset(C0, 0, sizeof(double) * M * N);
    memset(C1, 0, sizeof(double) * M * N);

    t0 = std::chrono::system_clock::now();
    fastnum::cpu::dgemm_tn(M, N, K, 0.5, A, M, B, N, 0, C0, N);
    t1 = std::chrono::system_clock::now();
    dgemm_naive_tn(M, N, K, 0.5, A, M, B, N, 0, C1, N);
    t2 = std::chrono::system_clock::now();

    duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    time0 = double(duration0.count()) / 1000000.0;
    time1 = double(duration1.count()) / 1000000.0;
    GFLOPS0 = (((2.0 * M * N / 1000) * K) / 1000000) / time0;
    GFLOPS1 = (((2.0 * M * N / 1000) * K) / 1000000) / time1;
    printf("dgemm_tn For input M:%d, N:%d, K=%d, time0:%f s, GFLOPS0:%f\n", M, N, K, time0, GFLOPS0);
    printf("dgemm_tn For input M:%d, N:%d, K=%d, time1:%f s, GFLOPS1:%f\n", M, N, K, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, N)) {
        printf("dgemm_tn Pass! Result Correctly!\n");
    } else {
        printf("dgemm_tn Fail! Result is Not Correctly!\n");
    }

    memset(C0, 0, sizeof(double) * M * N);
    memset(C1, 0, sizeof(double) * M * N);

    t0 = std::chrono::system_clock::now();
    fastnum::cpu::dgemm_nt(M, N, K, 0.5, A, K, B, K, 0, C0, N);
    t1 = std::chrono::system_clock::now();
    dgemm_naive_nt(M, N, K, 0.5, A, K, B, K, 0, C1, N);
    t2 = std::chrono::system_clock::now();

    duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    time0 = double(duration0.count()) / 1000000.0;
    time1 = double(duration1.count()) / 1000000.0;
    GFLOPS0 = (((2.0 * M * N / 1000) * K) / 1000000) / time0;
    GFLOPS1 = (((2.0 * M * N / 1000) * K) / 1000000) / time1;
    printf("dgemm_nt For input M:%d, N:%d, K=%d, time0:%f s, GFLOPS0:%f\n", M, N, K, time0, GFLOPS0);
    printf("dgemm_nt For input M:%d, N:%d, K=%d, time1:%f s, GFLOPS1:%f\n", M, N, K, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, N)) {
        printf("dgemm_nt Pass! Result Correctly!\n");
    } else {
        printf("dgemm_nt Fail! Result is Not Correctly!\n");
    }

    memset(C0, 0, sizeof(double) * M * N);
    memset(C1, 0, sizeof(double) * M * N);

    t0 = std::chrono::system_clock::now();
    fastnum::cpu::dgemm_tt(M, N, K, 0.5, A, M, B, K, 0, C0, N);
    t1 = std::chrono::system_clock::now();
    dgemm_naive_tt(M, N, K, 0.5, A, M, B, K, 0, C1, N);
    t2 = std::chrono::system_clock::now();

    duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    time0 = double(duration0.count()) / 1000000.0;
    time1 = double(duration1.count()) / 1000000.0;
    GFLOPS0 = (((2.0 * M * N / 1000) * K) / 1000000) / time0;
    GFLOPS1 = (((2.0 * M * N / 1000) * K) / 1000000) / time1;
    printf("dgemm_tt For input M:%d, N:%d, K=%d, time0:%f s, GFLOPS0:%f\n", M, N, K, time0, GFLOPS0);
    printf("dgemm_tt For input M:%d, N:%d, K=%d, time1:%f s, GFLOPS1:%f\n", M, N, K, time1, GFLOPS1);

    if (check_matrix(C0, C1, M, N)) {
        printf("dgemm_tt Pass! Result Correctly!\n");
    } else {
        printf("dgemm_tt Fail! Result is Not Correctly!\n");
    }

}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("the input args is not correct!\n");
        return 0;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    run_dgemm_test(M, N, K);
    return 0;
}

/*
.\dgemm_test.exe 1024 1024 1
.\dgemm_test.exe 1 1024 1024
.\dgemm_test.exe 1024 1 1024
*/