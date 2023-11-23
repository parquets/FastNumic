#include <stdio.h>
#include <string>
#include <chrono>
#include "matrix_utils.hpp"
#include "fastnum/im2col.hpp"


void test_im2col(int in_h, int in_w, int in_channels, int kernel_size) {
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    
    int out_h = (in_h-kernel_h)/stride_h + 1;
    int out_w = (in_w-kernel_w)/stride_w + 1;

    float* source = (float*)malloc(sizeof(float)*in_h*in_w*in_channels);
    float* dest = (float*)malloc(sizeof(float)*out_h*out_w*kernel_h*kernel_w*in_channels);

    random_matrix(source, in_h*in_w, in_channels);

    if(source == nullptr) {
        printf("malloc source fail!\n");
        return;
    }

    if(dest == nullptr) {
        printf("malloc dest fail!\n");
        return;
    }

    auto t0 = std::chrono::system_clock::now();
    fastnum::cpu::im2colPadFree(dest, source, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);
    auto t1 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    printf("im2col exec: %.3f micro second\n", (double)duration0.count());

    free(source);
    free(dest);
    
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("the input args is not correct!\n");
        return 0;
    }

    int in_h = atoi(argv[1]);
    int in_w = atoi(argv[2]);
    int in_channel = atoi(argv[3]);
    int kernel_size = atoi(argv[3]);

    test_im2col(in_h, in_w, in_channel, kernel_size);

    return 0;
}

/*
./im2col_test.exe 224 224 3 7
*/