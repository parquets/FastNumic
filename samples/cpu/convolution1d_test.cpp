#include <stdio.h>
#include <string>
#include <chrono>
#include "matrix_utils.hpp"
#include "convolution_naive.hpp"
#include "fastnum/convolution1d.hpp"

void test_convolution1d(int in_channels, int out_channels, int in_w, int kernel_w, int stride_w, int pad_w) {
    int out_w = (in_w - kernel_w) / stride_w + 1;

    float* input = (float*)malloc(sizeof(float) * in_channels * in_w);
    float* weight = (float*)malloc(sizeof(float) * out_channels * in_channels * in_w);

    random_matrix(input, in_channels, in_w);
    random_matrix(weight, in_channels*out_channels, kernel_w);


    float* output0 = (float*)malloc(sizeof(float) * out_channels * out_w);
    float* output1 = (float*)malloc(sizeof(float) * out_channels * out_w);

    memset(output0, 0, sizeof(float) * out_w * out_channels);
    memset(output1, 0, sizeof(float) * out_w * out_channels);

    auto t0 = std::chrono::system_clock::now();
    convolution1d_naive(output0, input, in_w, in_channels, out_channels, weight, kernel_w, stride_w, pad_w);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::convolution1d(output1, input, in_channels, out_channels, in_w, weight, kernel_w, stride_w, pad_w, 1);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;
    printf("convolution1d naive time = %.6f s\n", time0);
    printf("convolution1d avx time = %.6f s\n", time1);

    if(check_matrix(output0, output1, out_channels, out_w)) {
        printf("Pass! Result Correctly!\n");
    } else {
        printf("Fail! Result is Not Correctly!\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("the input args is not correct!\n");
        return 0;
    }

    int in_channel = atoi(argv[1]);
    int out_channel = atoi(argv[2]);
    int in_w = atoi(argv[3]);
    int kernel_w = atoi(argv[4]);
    int stride_w = atoi(argv[5]);

    test_convolution1d(in_channel, out_channel, in_w, kernel_w, stride_w, 0);

    return 0;
}

/*
./convolution1d_test.exe 64 8 1024 18 1
*/