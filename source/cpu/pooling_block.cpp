#include "pooling_block.hpp"

namespace fastnum {
namespace cpu {

void mean_pooling2d_row1(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w) {

    int out_w = (in_w - kernel_w) / stride_w + 1;

    int w = 0;
    for(w = 0; w < out_w - 31; w += 32) {
        kernel::meanPoolingHorizontal32(dest_value, 
                                        source + w * stride_w,
                                        in_w,
                                        kernel_h, kernel_w, stride_w);
        dest_value += 32;
    }
    for(; w < out_w - 15; w += 16) {
        kernel::meanPoolingHorizontal16(dest_value, 
                                        source + w * stride_w,
                                        in_w,
                                        kernel_h, kernel_w, stride_w);
        dest_value += 16;
    }

    for(; w < out_w - 7; w += 8) {
        kernel::meanPoolingHorizontal8(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 8;
    }
    for(; w < out_w - 3; w += 4) {
        kernel::meanPoolingHorizontal4(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 4;
    }
    for(; w < out_w; ++w) {
        kernel::meanPoolingHorizontal1(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 1;
    }
}

void max_pooling2d_row1(float* dest_value,
                        const float* source,
                        int in_channels,
                        int in_h, int in_w,
                        int kernel_h,
                        int kernel_w,
                        int stride_h,
                        int stride_w) {
    int out_w = (in_w - kernel_w) / stride_w + 1;

    int w = 0;
    for(w = 0; w < out_w - 31; w += 32) {
        kernel::maxPoolingHorizontal32(dest_value, 
                                        source + w * stride_w,
                                        in_w,
                                        kernel_h, kernel_w, stride_w);
        dest_value += 32;
    }
    for(; w < out_w - 15; w += 16) {
        kernel::maxPoolingHorizontal16(dest_value, 
                                        source + w * stride_w,
                                        in_w,
                                        kernel_h, kernel_w, stride_w);
        dest_value += 16;
    }

    for(; w < out_w - 7; w += 8) {
        kernel::maxPoolingHorizontal8(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 8;
    }
    for(; w < out_w - 3; w += 4) {
        kernel::maxPoolingHorizontal4(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 4;
    }
    for(; w < out_w; ++w) {
        kernel::maxPoolingHorizontal1(dest_value, 
                                       source + w * stride_w,
                                       in_w,
                                       kernel_h, kernel_w, stride_w);
        dest_value += 1;
    }
}


void mean_pooling2d_row4(float* dest_value,
                       const float* source,
                       int in_channels,
                       int in_h, int in_w,
                       int kernel_h,
                       int kernel_w,
                       int stride_h,
                       int stride_w) {
    int out_w = (in_w - kernel_w) / stride_w + 1;

    mean_pooling2d_row1(dest_value + 0 * out_w, 
                        source + 0 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    
    mean_pooling2d_row1(dest_value + 1 * out_w, 
                        source + 1 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);

    mean_pooling2d_row1(dest_value + 2 * out_w, 
                        source + 2 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    
    mean_pooling2d_row1(dest_value + 3 * out_w, 
                        source + 3 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
}

void max_pooling2d_row4(float* dest_value,
                       const float* source,
                       int in_channels,
                       int in_h, int in_w,
                       int kernel_h,
                       int kernel_w,
                       int stride_h,
                       int stride_w) {
    int out_w = (in_w - kernel_w) / stride_w + 1;

    max_pooling2d_row1(dest_value + 0 * out_w, 
                        source + 0 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    
    max_pooling2d_row1(dest_value + 1 * out_w, 
                        source + 1 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);

    max_pooling2d_row1(dest_value + 2 * out_w, 
                        source + 2 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    
    max_pooling2d_row1(dest_value + 3 * out_w, 
                        source + 3 * stride_w * in_w,
                        in_channels,
                        in_h, in_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
}


}
}