#include <stdio.h>
#include <string>
#include <chrono>

#include "matrix_utils.hpp"
#include "fastnum/pooling.hpp"


void mean_pooling2d(float* dest_value,
                  const float* source,
                  int in_channels,
                  int in_h, int in_w,
                  int kernel_h,
                  int kernel_w,
                  int stride_h,
                  int stride_w,
                  int pad_h,
                  int pad_w) {
    int karea = kernel_h * kernel_w;
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    for(int h=0; h < out_h; ++h) {
        for(int w=0; w < out_w; ++w) {
            int sh = h * stride_h;
            int sw = w * stride_w;

            float sum = 0;
            for(int kh =0; kh < kernel_h; ++kh) {
                for(int kw=0; kw < kernel_w; ++kw) {
                    sum += source[(sh + kh) * in_w + sw + kw];
                }
            }

            sum /= karea;
            dest_value[h * out_w + w] = sum;
        }
    }
}

void max_pooling2d(float* dest_value,
                   int* dest_index,
                  const float* source,
                  int in_channels,
                  int in_h, int in_w,
                  int kernel_h,
                  int kernel_w,
                  int stride_h,
                  int stride_w,
                  int pad_h,
                  int pad_w) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    for(int h=0; h < out_h; ++h) {
        for(int w=0; w < out_w; ++w) {
            int sh = h * stride_h;
            int sw = w * stride_w;
            float max_value = -FLT_MAX;
            int max_index = -1;
            for(int kh =0; kh < kernel_h; ++kh) {
                for(int kw=0; kw < kernel_w; ++kw) {
                    if(source[(sh + kh) * in_w + sw + kw] > max_value) {
                        max_value = source[(sh + kh) * in_w + sw + kw];
                        max_index = (sh + kh) * in_w + sw + kw;
                    }
                }
            }
            dest_value[h * out_w + w] = max_value;
            if(dest_index != nullptr) {
                dest_index[h * out_w + w] = max_index;
            }
        }
    }
}

void mean_pooling_test(int in_channels, int in_h, int in_w, int kernel_h, int kernel_w, int stride_h, int stride_w) {
    int pad_h = 0;
    int pad_w = 0;
    float* input = (float*)malloc(sizeof(float)*in_h * in_w * in_channels);
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    float* output0 = (float*)malloc(sizeof(float) * out_h * out_w * in_channels);
    float* output1 = (float*)malloc(sizeof(float) * out_h * out_w * in_channels);

    random_matrix(input, in_h * in_w , in_channels);
    memset(output0, 0, sizeof(float) * out_h * out_w * in_channels);
    memset(output1, 0, sizeof(float) * out_h * out_w * in_channels);

    auto t0 = std::chrono::system_clock::now();
    mean_pooling2d(output0, input, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, 0, 0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::mean_pooling2d(output1, input, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, 0, 0);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;
    printf("mean pooling2d naive time = %.6f s\n", time0);
    printf("mean pooling2d fastnum time = %.6f s\n", time1);

    if(check_matrix(output0, output1, in_channels, out_h * out_w)) {
        printf("Pass! Result is correct!\n");
    } else {
        printf("Fail! Result is not correct!\n");
    }
}

void max_pooling_test(int in_channels, int in_h, int in_w, int kernel_h, int kernel_w, int stride_h, int stride_w) {
    int pad_h = 0;
    int pad_w = 0;
    float* input = (float*)malloc(sizeof(float)*in_h * in_w * in_channels);
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    float* output0 = (float*)malloc(sizeof(float) * out_h * out_w * in_channels);
    float* output1 = (float*)malloc(sizeof(float) * out_h * out_w * in_channels);

    random_matrix(input, in_h * in_w , in_channels);
    memset(output0, 0, sizeof(float) * out_h * out_w * in_channels);
    memset(output1, 0, sizeof(float) * out_h * out_w * in_channels);

    auto t0 = std::chrono::system_clock::now();
    max_pooling2d(output0, nullptr, input, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, 0, 0);
    auto t1 = std::chrono::system_clock::now();
    fastnum::cpu::max_pooling2d(output1, input, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, 0, 0);
    auto t2 = std::chrono::system_clock::now();

    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double time0 = double(duration0.count()) / 1000000.0;
    double time1 = double(duration1.count()) / 1000000.0;
    printf("mean pooling2d naive time = %.6f s\n", time0);
    printf("mean pooling2d fastnum time = %.6f s\n", time1);

    if(check_matrix(output0, output1, in_channels, out_h * out_w)) {
        printf("Pass! Result is correct!\n");
    } else {
        printf("Fail! Result is not correct!\n");
    }
}

// void test_pooling(int in_channels, int in_w, int kernel_h, int kernel_w, int stride_h, int stride_w) {

// }



int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("the input args is not correct!\n");
        return 0;
    }

    int in_channel = atoi(argv[1]);
    int in_h = atoi(argv[2]);
    int in_w = atoi(argv[3]);
    int kernel_h = atoi(argv[4]);
    int kernel_w = atoi(argv[5]);
    int stride_h = atoi(argv[6]);
    int stride_w = atoi(argv[7]);

    // test_pooling(in_channel, in_w, kernel_w, stride_w, 0);
    mean_pooling_test(in_channel, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w);
    max_pooling_test(in_channel, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w);
    
    return 0;
}


/*
./pooling_test.exe 1 4 4 2 2 2 2
./pooling_test.exe 1 1023 1023 7 7 2 2
*/

