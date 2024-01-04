#pragma once

namespace fastnum {
namespace cpu {

void mean_pooling2d(float* dest_value,
                    const float* source,
                    int in_channels,
                    int in_h, int in_w,
                    int kernel_h,
                    int kernel_w,
                    int stride_h,
                    int stride_w,
                    int pad_h,
                    int pad_w);

void max_pooling2d(float* dest_value,
                   const float* source,
                   int in_channels,
                   int in_h, int in_w,
                   int kernel_h,
                   int kernel_w,
                   int stride_h,
                   int stride_w,
                   int pad_h,
                   int pad_w);

// void max_pooling2d(float* dest_value,
//                    int* dest_index,
//                    const float* source,
//                    int in_channels,
//                    int in_h, int in_w,
//                    int kernel_h,
//                    int kernel_w,
//                    int stride_h,
//                    int stride_w,
//                    int pad_h,
//                    int pad_w);


}
}