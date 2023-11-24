#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "fastnum/im2col.hpp"



namespace fastnum {
namespace cpu {
    
void im2colPadFreeDilationFree(float* dest, const float* source, 
                               int in_channels, int in_h, int in_w, 
                               int kernel_h, int kernel_w, 
                               int stride_h, int stride_w, 
                               int dilation_h, int dilation_w) {

    int out_h = (in_h-kernel_h) / stride_h + 1;
    int out_w = (in_w-kernel_w) / stride_w + 1;
    

    int c_stride = in_h*in_w;
    int col_stride = kernel_h*kernel_w*in_channels;
    int kernel_stride = kernel_h*kernel_w;

    for(int c=0; c<in_channels; ++c) {
        for(int oh=0; oh<out_h; ++oh) {
            for(int ow=0; ow<out_w; ++ow) {
                int ih_start = oh * stride_h;
                int iw_start = ow * stride_w;

                const float* s_ptr = source + ih_start * in_w + iw_start + c * c_stride;
                float* d_ptr = dest + (oh*out_w + ow)*col_stride + c * kernel_stride;

                for(int kh=0; kh<kernel_h; ++kh) {
                    memcpy(d_ptr, s_ptr + kh*in_w, sizeof(float)*kernel_w);
                    d_ptr += kernel_w;
                }

            }
        }
    }
}

void im2colPadFree(float* dest, const float* source, 
                   int in_channels, int in_h, int in_w, 
                   int kernel_h, int kernel_w, 
                   int stride_h, int stride_w, 
                   int dilation_h, int dilation_w) {

    int out_h = (in_h-kernel_h) / stride_h + 1;
    int out_w = (in_w-kernel_w) / stride_w + 1;
    

    int c_stride = in_h*in_w;
    int col_stride = kernel_h*kernel_w*in_channels;
    int kernel_stride = kernel_h*kernel_w;

    for(int c=0; c<in_channels; ++c) {
        for(int oh=0; oh<out_h; ++oh) {
            for(int ow=0; ow<out_w; ++ow) {
                int ih_start = oh * stride_h;
                int iw_start = ow * stride_w;

                const float* s_ptr = source + ih_start * in_w + iw_start + c * c_stride;
                float* d_ptr = dest + (oh*out_w + ow)*col_stride + c * kernel_stride;

                for(int kh=0; kh<kernel_h; ++kh) {
                    for(int kw=0; kw<kernel_w; ++kw) {
                        *d_ptr++ = s_ptr[kh * dilation_h * in_w + kw * dilation_w];
                    }
                }

            }
        }
    }
}



}
}