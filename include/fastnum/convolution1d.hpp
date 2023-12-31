#pragma once

namespace fastnum {
namespace cpu {

void convolution1d(float* output, 
                   const float* input, 
                   int in_channels, int out_channels, int in_w, 
                   const float* weight, int kernel_w, 
                   int stride_w, int pad_w, int dilation_w);

}
}