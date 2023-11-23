#include <stdio.h>
#include <string>
#include <chrono>
#include "convolution_naive.hpp"
#include "matrix_utils.hpp"
#include "fastnum/convolution.hpp"

void test_convolution(int in_h, int in_w, int in_channels, int out_channels) {

    float* src = (float*)malloc(sizeof(float)*in_h*in_w*in_channels);
    float* weight = (float*)malloc(sizeof(float)*3*3*in_channels*out_channels);

    // for(int ic=0; ic<in_channels; ++ic) {
    //     for(int i=0; i<in_h*in_w; ++i) src[ic*in_h*in_w + i] = i/10.0;
    // }

    // for(int c=0; c<out_channels*in_channels; ++c) {
    //     for(int i=0; i<3*3; ++i) weight[c*3*3 + i] = i/20.0;
    // }
    

    random_matrix(src, in_channels, in_h*in_w);
    random_matrix(weight, in_channels*out_channels, 3*3);
    

    int out_h = in_h-2;
    int out_w = in_w-2;
    
    float* dst0 = (float*)malloc(sizeof(float)*out_h*out_w*out_channels);
    float* dst1 = (float*)malloc(sizeof(float)*out_h*out_w*out_channels);

    memset(dst0, 0, sizeof(float) * out_h * out_w*out_channels);
    memset(dst1, 0, sizeof(float) * out_h * out_w*out_channels);

    auto t0 = std::chrono::system_clock::now();
    convolution_naive(dst0, src, in_h, in_w, in_channels, out_channels, weight, 3, 3, 1, 1, 0, 0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::winogradConv2dK3S1(dst1, src, in_channels, out_channels, in_h, in_w, weight);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;

    printf("convolution naive time = %.6f s\n", time0);
    printf("convolution winograd time = %.6f s\n", time1);

    double GFLOPS = ((((2.0*out_h*out_w*out_channels)/1000) * 3*3*in_channels)/1000000)/time1;

    printf("winograd conv GFLOPS = %.3f\n", GFLOPS);
    
    if (check_matrix(dst0, dst1, out_channels, out_h*out_w))
        printf("Pass! Result Correctly!\n");
    else
        printf("Fail! Result is Not Correctly!\n");
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

    test_convolution(in_h, in_w, in_channel, out_channel);

    return 0;
}
