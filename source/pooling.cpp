#include <math.h>
#include <stdlib.h>
#include "fastnum/pooling.hpp"
#include "cpu/pooling_block.hpp"


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
                    int pad_w) {
    
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;

    int h = 0;
    for(h = 0; h < out_h - 3; h += 4) {
        mean_pooling2d_row4(dest_value, source, 
                            in_channels, 
                            in_h, in_w, 
                            kernel_h, kernel_w, 
                            stride_h, stride_w);
        dest_value += 4 * out_h;
        source     += 4 * stride_h * in_w;
    }

    for(; h < out_h; ++h) {
        mean_pooling2d_row1(dest_value, source, 
                            in_channels, 
                            in_h, in_w, 
                            kernel_h, kernel_w, 
                            stride_h, stride_w);
        dest_value += out_h;
        source     += stride_h * in_w;
    }
}


void max_pooling2d(float* dest_value,
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

    int h = 0;
    for(h = 0; h < out_h - 3; h += 4) {
        max_pooling2d_row4(dest_value, source, 
                            in_channels, 
                            in_h, in_w, 
                            kernel_h, kernel_w, 
                            stride_h, stride_w);
        dest_value += 4 * out_h;
        source     += 4 * stride_h * in_w;
    }

    for(; h < out_h; ++h) {
        max_pooling2d_row1(dest_value, source, 
                            in_channels, 
                            in_h, in_w, 
                            kernel_h, kernel_w, 
                            stride_h, stride_w);
        dest_value += out_h;
        source     += stride_h * in_w;
    }
}

} // namespace cpu
} // namespace fastnum
