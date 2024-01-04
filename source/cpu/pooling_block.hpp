#pragma once

#include <math.h>
#include <float.h>
#include "x86/pooling_x86.hpp"

namespace fastnum {
namespace cpu {

namespace kernel {

static inline void meanPoolingHorizontal1(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {
    int karea = kernel_h * kernel_w;
    float sum = 0.0f;
    for(int kh=0; kh < kernel_h; ++kh) {
        for(int kw =0; kw < kernel_w; ++kw) {
            sum += source[kh * source_ldh + kw];
        }
    }
    dest_value[0] = sum/karea;
}

static inline void maxPoolingHorizontal1(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {
    int karea = kernel_h * kernel_w;
    float max_value = -FLT_MAX;
    for(int kh=0; kh < kernel_h; ++kh) {
        for(int kw =0; kw < kernel_w; ++kw) {
            max_value = std::max(source[kh * source_ldh + kw], max_value);
        }
    }
    dest_value[0] = max_value;
}

}

void mean_pooling2d_row1(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w);

void max_pooling2d_row1(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w);

void mean_pooling2d_row4(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w);

void max_pooling2d_row4(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w);

}
}