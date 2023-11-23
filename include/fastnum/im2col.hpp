#pragma once

namespace fastnum {
namespace cpu {

void im2colPadFree(float* dest, float* source, 
                   int in_channels, int in_h, int in_w, 
                   int kernel_h, int kernel_w, 
                   int stride_h, int stride_w, 
                   int dilation_h=1, int dilation_w=1);


}
}