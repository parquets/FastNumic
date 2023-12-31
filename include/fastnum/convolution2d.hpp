#pragma once

namespace fastnum {
namespace cpu {

void im2colPadFree();
void im2col();


void convolution2d(float* output, const float* input, int in_channels, int out_channels, int input_h, int input_w, 
                   const float* weight, int kernel_h, int kernel_w, 
                   int stride_h, int stride_w, int pad_h, int pad_w, 
                   int dilation, int groups);

void winogradConv2dK3S1(float* output, const float* input,
                        int in_channels, int out_channels,
                        int in_h, int in_w,
                        const float* weight);

void im2colGemmConv(float* output, const float* input,
                    const float* weight, 
                    int in_channels, int out_channels,
                    int in_h, int in_w,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h=0, int pad_w=0,
                    int dilation_h=1, int dilation_w=1);

void conv2dK1S1(float* dst, const float* src, const float* weight, int src_h, int src_w, int in_channels, int out_channels);
    
} // namespace cpu
} // namespace fastnum