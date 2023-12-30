#include <math.h>
#include "fastnum/convolution1d.hpp"
#include "cpu/x86/convolution1d_x86.hpp"


namespace fastnum {
namespace cpu {



void convolution1d(float* output, const float* input, int in_channels, int out_channels, int input_length, 
                   const float* weight, int kernel_w, 
                   int stride_w, int pad_w, int dilation_w) {
    
    
}


}
}