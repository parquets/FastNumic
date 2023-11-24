#include <stdio.h>
#include <string>
#include <chrono>
#include "convolution_naive.hpp"
#include "matrix_utils.hpp"
#include "fastnum/convolution.hpp"

void test_convolution(int in_h, int in_w, 
                      int in_channels, int out_channels, 
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w) {
    
    float* src = (float*)malloc(sizeof(float)*in_h*in_w*in_channels);
    float* weight = (float*)malloc(sizeof(float)*kernel_h*kernel_w*in_channels*out_channels);

    random_matrix(src, in_channels, in_h*in_w);
    random_matrix(weight, in_channels*out_channels, kernel_h*kernel_w);

    // for(int ic=0; ic<in_channels; ++ic) {
    //     float* src_ptr = src + ic*in_h*in_w;
    //     printf("------------channel:%d-----------\n", ic);
    //     for(int h=0; h<in_h; ++h) {
    //         for(int w=0; w<in_w; ++w) {
    //             printf("%.5f ", src_ptr[h*in_w+w]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n \n");
    // }

    int out_h = (in_h - kernel_h)/stride_h + 1;
    int out_w = (in_w - kernel_w)/stride_w + 1;

    float* dst0 = (float*)malloc(sizeof(float)*out_h*out_w*out_channels);
    float* dst1 = (float*)malloc(sizeof(float)*out_h*out_w*out_channels);

    memset(dst0, 0, sizeof(float) * out_h * out_w*out_channels);
    memset(dst1, 0, sizeof(float) * out_h * out_w*out_channels);

    auto t0 = std::chrono::system_clock::now();
    convolution_naive(dst0, src, in_h, in_w, in_channels, out_channels, weight, kernel_h, kernel_h, stride_h, stride_w, 0, 0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::im2colGemmConv(dst1, src, weight, in_channels, out_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;

    printf("convolution naive time = %.6f s\n", time0);
    printf("convolution winograd time = %.6f s\n", time1);

    double GFLOPS = ((((2.0*out_h*out_w*out_channels)/1000) * 3*3*in_channels)/1000000)/time1;
    printf("im2col conv GFLOPS = %.3f\n", GFLOPS);

     if (check_matrix(dst0, dst1, out_channels, out_h*out_w)) {
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

    int in_h = atoi(argv[1]);
    int in_w = atoi(argv[2]);
    int in_channel = atoi(argv[3]);
    int out_channel = atoi(argv[4]);
    int kernel_h = atoi(argv[5]);
    int kernel_w = atoi(argv[6]);
    int stride_h = atoi(argv[7]);
    int stride_w = atoi(argv[8]);

    test_convolution(in_h, in_w, in_channel, out_channel, kernel_h, kernel_w, stride_h, stride_w);

    return 0;
}

/*
./convolution_im2col_test.exe 224 224 3 64 7 7 2 2
./convolution_im2col_test.exe 112 112 64 128 3 3 1 1
./convolution_im2col_test.exe 152 202 128 256 3 3 1 1
./convolution_im2col_test.exe 3 3 1 1 3 3 1 1
*/