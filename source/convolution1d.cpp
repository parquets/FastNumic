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
                                        weight + oc * 8 * kernel_w, 
                                        in_channels, out_channels, kernel_w);

        // for(int x=0; x < 8 * in_channels * kernel_w; ++x) {
        //     printf("%.5f\n", packed_weight[x]);
        // }

        kernel::conv1dOCUnit8(output + oc * out_w, 
                              input, 
                              packed_weight, 
                              in_channels, out_channels,
                              in_w, kernel_w, stride_w, dilation_w);
    }

    free(packed_weight);

}


}
}