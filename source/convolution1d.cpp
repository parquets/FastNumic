#include <math.h>
#include "fastnum/convolution1d.hpp"
#include "cpu/x86/convolution1d_x86.hpp"


namespace fastnum {
namespace cpu {


void convolution1d(float* output, 
                   const float* input, 
                   int in_channels, int out_channels, int in_w, 
                   const float* weight, int kernel_w, 
                   int stride_w, int pad_w, int dilation_w) {


    int out_w = (in_w - kernel_w) / stride_w + 1;
    
    float* packed_weight = (float*)malloc(sizeof(float) * 8 * in_channels * kernel_w);

    int oc = 0;
    for(oc = 0; oc < out_channels - 7; oc += 8) {
        kernel::conv1dWeightPackOCUnit8(packed_weight, 
                                        weight, 
                                        in_channels, out_channels, kernel_w);
        
        kernel::conv1dOCUnit8(output, 
                              input, 
                              packed_weight, 
                              in_channels, out_channels,
                              in_w, kernel_w, stride_w, dilation_w);

        weight += 8 * in_channels * kernel_w;
        output += 8 * out_w;
    }

    for(; oc < out_channels - 3; oc += 4) {
        kernel::conv1dWeightPackOCUnit4(packed_weight,
                                        weight,
                                        in_channels, out_channels, kernel_w);
        
        kernel::conv1dOCUnit4(output,
                              input,
                              packed_weight,
                              in_channels, out_channels,
                              in_w, kernel_w, stride_w, dilation_w);
        
        weight += 4 * in_channels * kernel_w;;
        output += 4 * out_w;
    }
    for(; oc < out_channels; ++oc) {
        kernel::conv1dWeightPackOCUnit1(packed_weight,
                                        weight,
                                        in_channels, out_channels, kernel_w);
        
        kernel::conv1dOCUnit1(output,
                              input,
                              packed_weight,
                              in_channels, out_channels,
                              in_w, kernel_w, stride_w, dilation_w);
        
        weight += in_channels * kernel_w;
        output += out_w;
    }
    free(packed_weight);

}


}
}