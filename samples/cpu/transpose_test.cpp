#include <stdio.h>
#include <math.h>
#include <chrono>
#include "fastnum/transpose.hpp"
#include "matrix_utils.hpp"
#include "transpose_naive.hpp"

int run_transpose_test(int M, int N) {
    float* fA = (float*)malloc(sizeof(float)*M*N);
    float* fB0 = (float*)malloc(sizeof(float)*M*N);
    float* fB1 = (float*)malloc(sizeof(float)*M*N);

    double* dA = (double*)malloc(sizeof(double)*M*N);
    double* dB0 = (double*)malloc(sizeof(double)*M*N);
    double* dB1 = (double*)malloc(sizeof(double)*M*N);

    random_matrix(fA, M, N);
    random_matrix(dA, M, N);

    auto t0 = std::chrono::system_clock::now();
    transpose_naive(fA, M, N, N, fB0, M);
    auto t1 = std::chrono::system_clock::now();
    transpose_naive(dA, M, N, N, dB0, M);

    auto t2 = std::chrono::system_clock::now();
    fastnum::cpu::transpose(M, N, fA, N, fB1, M);
    auto t3 = std::chrono::system_clock::now();
    fastnum::cpu::transpose(M, N, dA, N, dB1, M);
    auto t4 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
    double time0 = double(duration0.count());
    double time1 = double(duration1.count());
    double time2 = double(duration2.count());
    double time3 = double(duration3.count());

    if(check_matrix(fB0, fB1, N, M)) {
        printf("float transpose correct!\n");
    }

    if(check_matrix(dB0, dB1, N, M)) {
        printf("double transpose correct!\n");
    }

    printf("float transpose navie time = %.3f micro second, fastnum time = %.3f micro second\n", time0, time1);
    printf("double transpose navie time = %.3f micro second, fastnum time = %.3f micro second\n", time2, time3);

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