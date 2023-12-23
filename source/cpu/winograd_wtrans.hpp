#include <math.h>
#include <memory.h>


namespace fastnum {
namespace cpu {


static inline void weightTransLeftF2(float* dest, const float* source) {
    dest[0] = source[0];
    dest[1] = source[1];
    dest[2] = source[2];
    dest[3] = (source[0] + source[3] + source[6]) / 2;
    dest[4] = (source[1] + source[4] + source[7]) / 2;
    dest[5] = (source[2] + source[5] + source[8]) / 2;
    dest[6] = (source[0] - source[3] + source[6]) / 2;
    dest[7] = (source[1] - source[4] + source[7]) / 2;
    dest[8] = (source[2] - source[5] + source[8]) / 2;
    dest[9] = source[6];
    dest[10] = source[7];
    dest[11] = source[8];
}

static inline void weightTransRightF2(float* dest, const float* source) {
    dest[0] = source[0];
    dest[1] = (source[0] + source[1] + source[2]) / 2;
    dest[2] = (source[0] - source[1] + source[2]) / 2;
    dest[3] = source[2];

    dest[4] = source[3];
    dest[5] = (source[3] + source[4] + source[5]) / 2;
    dest[6] = (source[3] - source[4] + source[5]) / 2;
    dest[7] = source[5];

    dest[8] = source[6];
    dest[9] = (source[6] + source[7] + source[8]) / 2;
    dest[10] = (source[6] - source[7] + source[8]) / 2;
    dest[11] = source[8];

    dest[12] = source[9];
    dest[13] = (source[9] + source[10] + source[11]) / 2;
    dest[14] = (source[9] - source[10] + source[11]) / 2;
    dest[15] = source[11];
}





static inline void weightTransLeftF6(float* dest, const float* source)  {
    /*
        Gg -> [8, 3]
        g  -> [3, 3]
        */
        const float c0 = static_cast<float>(2.0 / 9);
        const float c1 = static_cast<float>(1.0 / 90);
        const float c2 = static_cast<float>(1.0 / 45);
        const float c3 = static_cast<float>(2.0 / 45);
        const float c4 = static_cast<float>(1.0 / 180);

        dest[0 * 3 + 0] = source[0];
        dest[0 * 3 + 1] = source[1];
        dest[0 * 3 + 2] = source[2];

        dest[1 * 3 + 0] = -c0 * (source[0] + source[3] + source[6]);
        dest[1 * 3 + 1] = -c0 * (source[1] + source[4] + source[7]);
        dest[1 * 3 + 2] = -c0 * (source[2] + source[5] + source[8]);

        dest[2 * 3 + 0] = -c0 * (source[0] - source[3] + source[6]);
        dest[2 * 3 + 1] = -c0 * (source[1] - source[4] + source[7]);
        dest[2 * 3 + 2] = -c0 * (source[2] - source[5] + source[8]);

        dest[3 * 3 + 0] = c1 * source[0] + c2 * source[3] + c3 * source[6];
        dest[3 * 3 + 1] = c1 * source[1] + c2 * source[4] + c3 * source[7];
        dest[3 * 3 + 2] = c1 * source[2] + c2 * source[5] + c3 * source[8];

        dest[4 * 3 + 0] = c1 * source[0] - c2 * source[3] + c3 * source[6];
        dest[4 * 3 + 1] = c1 * source[1] - c2 * source[4] + c3 * source[7];
        dest[4 * 3 + 2] = c1 * source[2] - c2 * source[5] + c3 * source[8];

        dest[5 * 3 + 0] = c2 * source[0] + c1 * source[3] + c4 * source[6];
        dest[5 * 3 + 1] = c2 * source[1] + c1 * source[4] + c4 * source[7];
        dest[5 * 3 + 2] = c2 * source[2] + c1 * source[5] + c4 * source[8];

        dest[6 * 3 + 0] = c2 * source[0] - c1 * source[3] + c4 * source[6];
        dest[6 * 3 + 1] = c2 * source[1] - c1 * source[4] + c4 * source[7];
        dest[6 * 3 + 2] = c2 * source[2] - c1 * source[5] + c4 * source[8];

        dest[7 * 3 + 0] = source[6];
        dest[7 * 3 + 1] = source[7];
        dest[7 * 3 + 2] = source[8];
}

static inline void weightTransRightF6(float* dest, const float* source) {
    const float c0 = static_cast<float>(2.0 / 9);
    const float c1 = static_cast<float>(1.0 / 90);
    const float c2 = static_cast<float>(1.0 / 45);
    const float c3 = static_cast<float>(2.0 / 45);
    const float c4 = static_cast<float>(1.0 / 180);

    for (int i = 0; i < 8; ++i) {
        int i0 = i * 3;
        dest[i * 8 + 0] = source[i0];
        dest[i * 8 + 1] = -c0 * (source[i0] + source[i0 + 1] + source[i0 + 2]);
        dest[i * 8 + 2] = -c0 * (source[i0] - source[i0 + 1] + source[i0 + 2]);
        dest[i * 8 + 3] = c1 * source[i0] + c2 * source[i0 + 1] + c3 * source[i0 + 2];
        dest[i * 8 + 4] = c1 * source[i0] - c2 * source[i0 + 1] + c3 * source[i0 + 2];
        dest[i * 8 + 5] = c2 * source[i0] + c1 * source[i0 + 1] + c4 * source[i0 + 2];
        dest[i * 8 + 6] = c2 * source[i0] - c1 * source[i0 + 1] + c4 * source[i0 + 2];
        dest[i * 8 + 7] = source[i0 + 2];
    }
}


void winogradWeightTransK3S1F6(float* dest, float* source,
                              int in_channels, int out_channels) {
    
    memset(dest, 0, sizeof(float) * (out_channels+3)/4 * 8 * 8);

    float source_tmp[4*3*3] = {0};
    
    int karea = 3*3;
    int ch_stride = karea*in_channels;

    for(int ic = 0; ic < in_channels; ++ic) {       
        for(int oc=0; oc < out_channels - 3; oc += 4) {

            for(int kh=0; kh < 3; ++kh) {
                for(int kw=0; kw < 3; ++kw) {
                    source_tmp[kh*3*4+kw*4 + 0] = source[0*karea + kh*3 + kw];
                    source_tmp[kh*3*4+kw*4 + 1] = source[1*karea + kh*3 + kw];
                    source_tmp[kh*3*4+kw*4 + 2] = source[2*karea + kh*3 + kw];
                    source_tmp[kh*3*4+kw*4 + 3] = source[3*karea + kh*3 + kw];
                }
            }      
        }
    }
}

}
}