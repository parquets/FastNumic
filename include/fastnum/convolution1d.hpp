#pragma once

namespace fastnum {
namespace cpu {

void convolution1d(float* output, const float* input, int in_channels, int out_channels, int input_length, 
                   const float* weight, int kernel, 
                   int stride, int pad, int dilation, int group);

}
}