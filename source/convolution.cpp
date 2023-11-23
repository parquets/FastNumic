#include <algorithm>
#include <chrono>
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

static void im2colGemmConv(float* output, const float* input,
                    int in_channels, int out_channels,
                    int in_h, int in_w,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w) {
    
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

    float* u_buffer = (float*)malloc(sizeof(float)*8*8*8*4);
    float* v_buffer = (float*)malloc(sizeof(float)*nh6*nw6*8*8*8);
    // float* uv_buffer = (float*)malloc(sizeof(float)*in_h*in_w*8*8*8);


    int ic = 0;

    for(ic=0; ic < in_channels - 7; ic += 8) {
        winogradDataTransSliceUnit8K3S1Pack8(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
        for(int oc=0; oc < out_channels; ++oc) {
            kernel::winogradWeightTransUnit8K3S1Pack8(u_buffer, weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
        
            winogradUVTransSliceUnit8K3S1Pack8(output + oc * out_ch_stride, 
                                               dest_cols, 
                                               v_buffer, 
                                               u_buffer,
                                               in_h, in_w, 
                                               in_w, in_h*in_w);
        }
    }

    for(; ic < in_channels - 3; ic += 4) {
        winogradDataTransSliceUnit8K3S1Pack4(v_buffer, input + ic*in_ch_stride, in_h, in_w, in_w, in_w*in_h);
        for(int oc=0; oc < out_channels; ++oc) {
            kernel::winogradWeightTransUnit8K3S1Pack4(u_buffer, weight + oc*k_ch_stride*in_channels + ic * k_ch_stride);
            winogradUVTransSliceUnit8K3S1Pack4(output + oc * out_ch_stride, dest_cols, 
                                               v_buffer, u_buffer, 
                                               in_h, in_w, 
                                               in_w, in_h*in_w);
        }
    }

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

    // int in_ch=0;

    // for(in_ch=0; in_ch < in_channels-7; in_ch += 8) {
    //     winogradConvK3S1Pack8(output, 
    //                           u_buffer, 
    //                           v_buffer, 
    //                           uv_buffer, 
    //                           input + in_ch * in_ch_stride, 
    //                           weight + in_ch * k_ch_stride, 
    //                           in_h, in_w);
    // }

    // for(; in_ch < in_channels-3; in_ch += 4) {
    //     winogradConvK3S1Pack4(output, 
    //                           u_buffer, 
    //                           v_buffer, 
    //                           uv_buffer, 
    //                           input + in_ch * in_ch_stride, 
    //                           weight + in_ch * k_ch_stride, 
    //                           in_h, in_w);
    // }

    // for(; in_ch < in_channels; ++in_ch) {
    //     winogradConvK3S1Pack1(output, 
    //                           u_buffer, 
    //                           v_buffer, 
    //                           uv_buffer, 
    //                           input + in_ch * in_ch_stride, 
    //                           weight + in_ch * k_ch_stride, 
    //                           in_h, in_w);
    // }



    free(u_buffer);
    free(v_buffer);
    // free(uv_buffer);
}

// void winogradConv2dK3S1(float* output, const float* input,
//                         int in_channels, int out_channels,
//                         int in_h, int in_w,
//                         const float* weight) {
    
//     int out_h = in_h - 2;
//     int out_w = in_w - 2;
//     int out_ch_stride = out_h*out_w;
//     int w_stride = 3*3 * in_channels;

//     int oc = 0;

//     for(int oc = 0; oc<out_channels; ++oc) {
//         winogradConv2dOC1K3S1(output + oc * out_ch_stride,
//                               input,
//                               in_channels,
//                               in_h, in_w,
//                               weight + oc*w_stride);
//     }
// }

}
}