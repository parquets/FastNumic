#include <algorithm>
#include <chrono>
#include "fastnum/im2col.hpp"
#include "fastnum/gemm.hpp"
#include "fastnum/convolution.hpp"
#include "cpu/winograd_block.hpp"
#include "cpu/x86/arithmetic_x86.hpp"
#include "cpu/x86/convolution_k1x1_x86.hpp"


namespace fastnum {
namespace cpu {

void conv2dK1S1(float* dst, const float* src, const float* weight, int src_h, int src_w, int in_channels, int out_channels) {
    int ch_stride = src_h * src_w;
    for (int oc = 0; oc < out_channels; ++oc) {
        int ic = 0;
        for (ic = 0; ic < in_channels - 5; ic += 6) {
            kernel::conv2dK1S1Ch6(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
        for (; ic < in_channels - 3; ic += 4) {
            kernel::conv2dK1S1Ch4(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
        for (; ic < in_channels; ++ic) {
            kernel::conv2dK1S1Ch1(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
    }
}

void im2colGemmConv(float* output, const float* input,
                    const float* weight, 
                    int in_channels, int out_channels,
                    int in_h, int in_w,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dilation_h, int dilation_w) {

    int out_h = (in_h-kernel_h+2*pad_h) / stride_h + 1;
    int out_w = (in_w-kernel_w+2*pad_w) / stride_w + 1;

    int M = out_channels;
    int N = out_h * out_w;
    int K = kernel_h*kernel_w*in_channels;

    float* coldata = (float*)malloc(sizeof(float)*K*N);

    im2colPadFreeDilationFree(coldata, input, in_channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);

    sgemm_nn(M, N, K, 1.0, weight, K, coldata, N, 0.0, output, N);
}

void winogradConv2dK3S1(float* output, const float* input,
                        int in_channels, int out_channels,
                        int in_h, int in_w,
                        const float* weight) {

    int in_ch_stride = in_h*in_w;
    int k_ch_stride = 3*3;
    int out_ch_stride = (in_h-2)*(in_w-2);

    int nh6 = (in_h-2)/6;
    int nw6 = (in_w-2)/6;

    int dest_cols = in_w - 2;
    int dest_rows = in_h - 2;
    float* u_buffer = (float*)malloc(sizeof(float)*8*8*8*out_channels);
    float* v_buffer = (float*)malloc(sizeof(float)*nh6*nw6*8*8*8);
    // float* uv_buffer = (float*)malloc(sizeof(float)*in_h*in_w*8*8*8);

    float* output_ptr = nullptr;
    float* u_buffer_ptr = nullptr;
    float* v_buffer_ptr = nullptr;
    
    int ic = 0;

    for(ic=0; ic < in_channels - 7; ic += 8) {
        winogradDataTransSliceUnit8K3S1Pack8(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
        for(int oc = 0; oc < out_channels; ++oc) {
            kernel::winogradWeightTransUnit8K3S1Pack8(u_buffer + oc*8*8*8, 
                                                     weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
        }

        int h=0, w=0;
        for(h = 0; h < in_h - 7; h += 6) {
            u_buffer_ptr = u_buffer;
            for(int oc=0; oc < out_channels; ++oc) {
                output_ptr = output + oc * out_ch_stride + h*dest_cols;
                v_buffer_ptr = v_buffer + (h/6)*nw6*8*8*8;
                for(w = 0; w < in_w - 7; w += 6) {
                    kernel::winogradUVTransUnit8K3S1Pack8(output_ptr + w, 
                                                          dest_cols, 
                                                          v_buffer_ptr, 
                                                          u_buffer_ptr, 
                                                          8*8);
                    v_buffer_ptr += 8*8*8;
                }
                u_buffer_ptr += 8*8*8;
            }
            
        }
    }
    // for(ic=0; ic < in_channels - 7; ic += 8) {
    //     winogradDataTransSliceUnit8K3S1Pack8(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
    //     for(int oc=0; oc < out_channels-1; oc+=2) {
    //         kernel::winogradWeightTransUnit8K3S1Pack8(u_buffer+0*8*8*8, weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
    //         kernel::winogradWeightTransUnit8K3S1Pack8(u_buffer+1*8*8*8, weight + (oc+1)*k_ch_stride*in_channels + ic * k_ch_stride);
        
    //         winogradUVTransSliceUnit8K3S1Pack8(output + oc * out_ch_stride, 
    //                                            dest_cols,
    //                                            v_buffer, 
    //                                            u_buffer,
    //                                            in_h, in_w, 
    //                                            in_w, in_h*in_w);
    //         winogradUVTransSliceUnit8K3S1Pack8(output + (oc+1) * out_ch_stride, 
    //                                            dest_cols,
    //                                            v_buffer, 
    //                                            u_buffer+8*8*8,
    //                                            in_h, in_w, 
    //                                            in_w, in_h*in_w);
    //     }
    // }


    // for(; ic < in_channels - 3; ic += 4) {
    //     winogradDataTransSliceUnit8K3S1Pack4(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
    //     for(int oc=0; oc < out_channels; ++oc) {
    //         kernel::winogradWeightTransUnit8K3S1Pack4(u_buffer, weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
    //         winogradUVTransSliceUnit8K3S1Pack4(output + oc * out_ch_stride, dest_cols, 
    //                                            v_buffer, u_buffer, 
    //                                            in_h, in_w, 
    //                                            in_w, in_h*in_w);
    //     }
    // }

    // for(; ic < in_channels; ++ic) {
    //     winogradDataTransSliceUnit8K3S1Pack1(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
    //     for(int oc=0; oc < out_channels; ++oc) {
    //         kernel::winogradWeightTransUnit8K3S1Pack1(u_buffer, weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
    //         winogradUVTransSliceUnit8K3S1Pack1(output + oc * out_ch_stride, dest_cols, 
    //                                            v_buffer, u_buffer, 
    //                                            in_h, in_w, 
    //                                            in_w, in_h*in_w);
    //     }
    // }




    free(u_buffer);
    free(v_buffer);
}





}
}