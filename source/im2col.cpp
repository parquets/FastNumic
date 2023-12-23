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

    int source_chan_stride = in_h * in_w;
    int cols = out_h*out_w;
    int karea = kernel_h*kernel_w;
    
    const float* source_ptr = source;
    
    for(int c = 0; c < in_channels; ++c) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                for(int h=0; h < out_h; ++h) {
                    for(int w=0; w < out_w; ++w) {
                        int offset_h = h * stride_h;
                        int offset_w = w * stride_w;
                        int d_ind = (c*karea + kh*kernel_w + kw)*cols + h*out_w + w;
                        int s_ind = c * source_chan_stride + (offset_h + kh) * in_w + offset_w + kw;
                        dest[d_ind] = source[s_ind];
                    }
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

    int source_chan_stride = in_h * in_w;
    int cols = out_h*out_w;
    int karea = kernel_h*kernel_w;
    
    const float* source_ptr = source;
    
    for(int c = 0; c < in_channels; ++c) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                int dkh = kh * dilation_h;
                int dkw = kw * dilation_w;
                for(int h=0; h < out_h; ++h) {
                    for(int w=0; w < out_w; ++w) {
                        int offset_h = h * stride_h;
                        int offset_w = w * stride_w;
                        int d_ind = (c*karea + kh*kernel_w + kw)*cols + h*out_w + w;
                        int s_ind = c * source_chan_stride + (offset_h + dkh) * in_w + offset_w + dkw;
                        dest[d_ind] = source[s_ind];
                    }
                }
            }
        }
    }
}



}
}