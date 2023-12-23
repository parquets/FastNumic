#include "winograd_block.hpp"


namespace fastnum {
namespace cpu {



void convK3S1NaivePack1(float* dest, const float* source, const float* weight, int h_stride, int ch_stride) {
    *dest += (weight[0]*source[0] + weight[1]*source[1] + weight[2]*source[2]);
    source += h_stride;
    weight += 3;
    *dest += (weight[0]*source[0] + weight[1]*source[1] + weight[2]*source[2]);
    source += h_stride;
    weight += 3;
    *dest += (weight[0]*source[0] + weight[1]*source[1] + weight[2]*source[2]);
}
void convK3S1NaivePack4(float* dest, const float* source, const float* weight, int h_stride, int ch_stride) {
    for(int ch=0; ch<4; ++ch) {
        convK3S1NaivePack1(dest, source, weight, h_stride, ch_stride);
        source += ch_stride;
        weight += 9;
    }
}
void convK3S1NaivePack8(float* dest, const float* source, const float* weight, int h_stride, int ch_stride) {
    for(int ch=0; ch<8; ++ch) {
        convK3S1NaivePack1(dest, source, weight, h_stride, ch_stride);
        source += ch_stride;
        weight += 9;
    }
}


void winogradConvK3S1Unit4Pack1(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest) {
    int dest_cols = ldest;
    kernel::winogradWeightTransUnit4K3S1Pack1(u_buffer, weight);

    int r=0, c=0;

    for(r=0; r<src_h - 3; r += 2) {
        for(c=0; c<src_w - 3; c += 2) {
            kernel::winogradDataTransUnit4K3S1Pack1(v_buffer, 4*1, source + c, ldh, ldc);
            kernel::winogradUVTransUnit4K3S1Pack1(dest + c, ldest, v_buffer, u_buffer, 4*1);
        }
        for(; c < src_w - 2; ++c) {
            convK3S1NaivePack1(dest+0*ldest+c, source+0*ldh+c, weight, ldh, ldc);
            convK3S1NaivePack1(dest+1*ldest+c, source+1*ldh+c, weight, ldh, ldc);
        }
        source += 2 * ldh;
        dest += 2 * ldest;
    }
    for(; r<src_h - 2; ++r) {
        for(c=0; c<src_w - 2; ++c) {
            convK3S1NaivePack1(dest+c, source+c, weight, ldh, ldc);
        }
    }
}

void winogradConvK3S1Unit4Pack4(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest) {

    kernel::winogradWeightTransUnit4K3S1Pack4(u_buffer, weight);

    int r=0, c=0;

    for(r=0; r < src_h - 3; r += 2) {
        for(c=0; c < src_w - 3; c += 2) {
            kernel::winogradDataTransUnit4K3S1Pack4(v_buffer, 4*4, source + c, ldh, ldc);
            kernel::winogradUVTransUnit4K3S1Pack4(dest + c, ldest, v_buffer, u_buffer, 4*4);
        }
        for(; c < src_w - 2; ++c) {
            convK3S1NaivePack4(dest+0*ldest+c, source+0*ldh+c, weight, ldh, ldc);
            convK3S1NaivePack4(dest+1*ldest+c, source+1*ldh+c, weight, ldh, ldc);
        }
        source += 2 * ldh;
        dest += 2 * ldest;
    }

    for(; r<src_h - 2; ++r) {
        for(c=0; c<src_w - 2; ++c) {
            convK3S1NaivePack4(dest+c, source+c, weight, ldh, ldc);
        }
    }
}

void winogradConvK3S1Unit4Pack8(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest) {
    
    kernel::winogradWeightTransUnit4K3S1Pack8(u_buffer, weight);

    int r=0, c=0;

    for(r=0; r < src_h - 3; r += 2) {
        for(c=0; c < src_w - 3; c += 2) {
            kernel::winogradDataTransUnit4K3S1Pack8(v_buffer, 4*8, source + c, ldh, ldc);
            kernel::winogradUVTransUnit4K3S1Pack8(dest + c, ldest, v_buffer, u_buffer, 4*8);
        }
    
        for(; c < src_w - 2; ++c) {
            convK3S1NaivePack8(dest+0*ldest+c, source+0*ldh+c, weight, ldh, ldc);
            convK3S1NaivePack8(dest+1*ldest+c, source+1*ldh+c, weight, ldh, ldc);
        }
        source += 2 * ldh;
        dest += 2 * ldest;
    }

    for(; r < src_h - 2; ++r) {
        for(c = 0; c < src_w - 2; ++c) {
            convK3S1NaivePack8(dest+c, source+c, weight, ldh, ldc);
        }
    }
}


// void winogradConvK3S1Unit8Pack1(float* dest, 
//                                 float* u_buffer, 
//                                 float* v_buffer, 
//                                 float* uv_buffer,
//                                 const float* source, 
//                                 int src_h, int src_w,
//                                 int ldh, int ldc,
//                                 const float* weight,
//                                 int ldest) {
//     kernel::winogradWeightTransUnit8K3S1Pack1(u_buffer, weight);

//     int r=0, c=0;

//     for(r=0; r<src_h - 7; r += 6) {
//         for(c=0; c<src_w - 7; c += 6) {
//             kernel::winogradDataTransUnit8K3S1Pack1(v_buffer, 8*1, source + c, ldh, ldc);
//             kernel::winogradUVTransUnit8K3S1Pack1(dest + c, ldest, v_buffer, u_buffer, 8*1);
//         }
//         source += 6 * ldh;
//         dest += 6 * ldest;
//     }
// }


void winogradConvK3S1Unit8Pack4(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest) {
    kernel::winogradWeightTransUnit8K3S1Pack4(u_buffer, weight);

    int r=0, c=0;

    for(r=0; r<src_h - 7; r += 6) {
        for(c=0; c<src_w - 7; c += 6) {
            kernel::winogradDataTransUnit8K3S1Pack4(v_buffer, 8*4, source + c, ldh, ldc);
            kernel::winogradUVTransUnit8K3S1Pack4(dest + c, ldest, v_buffer, u_buffer, 8*4);
        }
        source += 6 * ldh;
        dest += 6 * ldest;
    }
}


void winogradConvK3S1Unit8Pack8(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest) {
    kernel::winogradWeightTransUnit8K3S1Pack8(u_buffer, weight);

    double data_t = 0, uv_t = 0;

    int r=0, c=0;
    for(r=0; r<src_h - 7; r += 6) {
        for(c=0; c<src_w - 7; c += 6) {
            kernel::winogradDataTransUnit8K3S1Pack8(v_buffer, 8*8, source + c, ldh, ldc);
            kernel::winogradUVTransUnit8K3S1Pack8(dest + c, ldest, v_buffer, u_buffer, 8*8);
        }
        source += 6 * ldh;
        dest += 6 * ldest;
    }

}



void winogradConvK3S1Pack1(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols) {
    
    int nr6 = (int)(rows - 2.0) / 6;
    int nc6 = (int)(cols - 2.0) / 6;

    int nr4_offset = nr6*6;
    int nc4_offset = nc6*6;

    int nr4 = (rows - nr4_offset - 2) / 2; 
    int nc4 = (cols - nc4_offset - 2) / 2;

    int dest_cols = cols - 2;

    int r = 0, c = 0;

    winogradConvK3S1Unit8Pack1(dest, 
                               u_buffer, v_buffer, uv_buffer, 
                               source, 
                               rows, cols,
                               cols, rows*cols, 
                               weight, dest_cols);

    winogradConvK3S1Unit4Pack1(dest + nc4_offset, 
                               u_buffer, v_buffer, uv_buffer, 
                               source + nc4_offset, 
                               nr4_offset + 2, cols - nc4_offset,
                               cols, rows*cols,
                               weight, 
                               dest_cols);
    
    winogradConvK3S1Unit4Pack1(dest + nr4_offset * dest_cols,
                               u_buffer, v_buffer, uv_buffer,
                               source + nr4_offset * cols,
                               rows - nr4_offset, cols,
                               cols, rows*cols,
                               weight,
                               dest_cols);

}

void winogradConvK3S1Pack4(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols) {
    
    int nr6 = (int)(rows - 2.0) / 6;
    int nc6 = (int)(cols - 2.0) / 6;

    int nr4_offset = nr6*6;
    int nc4_offset = nc6*6;

    int nr4 = (rows - nr4_offset - 2) / 2; 
    int nc4 = (cols - nc4_offset - 2) / 2;

    int dest_cols = cols - 2;

    int r = 0, c = 0;
    
    winogradConvK3S1Unit8Pack4(dest, 
                               u_buffer, v_buffer, uv_buffer, 
                               source, 
                               rows, cols,
                               cols, rows*cols, 
                               weight, dest_cols);

    winogradConvK3S1Unit4Pack4(dest + nc4_offset, 
                               u_buffer, v_buffer, uv_buffer, 
                               source + nc4_offset, 
                               nr4_offset + 2, cols - nc4_offset,
                               cols, rows*cols,
                               weight, 
                               dest_cols);
    
    winogradConvK3S1Unit4Pack4(dest + nr4_offset * dest_cols,
                               u_buffer, v_buffer, uv_buffer,
                               source + nr4_offset * cols,
                               rows - nr4_offset, cols,
                               cols, rows*cols,
                               weight,
                               dest_cols);

}

void winogradConvK3S1Pack8(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols) {
    
    int nr6 = (int)(rows - 2.0) / 6;
    int nc6 = (int)(cols - 2.0) / 6;

    int nr4_offset = nr6*6;
    int nc4_offset = nc6*6;

    int nr4 = (rows - nr4_offset - 2) / 2; 
    int nc4 = (cols - nc4_offset - 2) / 2;

    int dest_cols = cols - 2;

    int r = 0, c = 0;

    winogradConvK3S1Unit8Pack8(dest, 
                               u_buffer, v_buffer, uv_buffer, 
                               source, 
                               rows, cols,
                               cols, rows*cols, 
                               weight, dest_cols);

    winogradConvK3S1Unit4Pack8(dest + nc4_offset, 
                               u_buffer, v_buffer, uv_buffer, 
                               source + nc4_offset, 
                               nr4_offset + 2, cols - nc4_offset,
                               cols, rows*cols,
                               weight, 
                               dest_cols);
    
    winogradConvK3S1Unit4Pack8(dest + nr4_offset * dest_cols,
                               u_buffer, v_buffer, uv_buffer,
                               source + nr4_offset * cols,
                               rows - nr4_offset, cols,
                               cols, rows*cols,
                               weight,
                               dest_cols);

}

void winogradDataTransSliceUnit8K3S1Pack1(float* dest, 
                                          const float* source, 
                                          int in_h, int in_w, 
                                          int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradDataTransUnit8K3S1Pack1(dest, 8*1, source + w, ldh, ldc);
            dest += 8*8*1;
        }
        source += 6 * ldh;
    }
}

void winogradDataTransSliceUnit8K3S1Pack4(float* dest, 
                                          const float* source, 
                                          int in_h, int in_w, 
                                          int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradDataTransUnit8K3S1Pack4(dest, 8*4, source + w, ldh, ldc);
            dest += 8*8*4;
        }
        source += 6 * ldh;
    }
}

void winogradDataTransSliceUnit8K3S1Pack8(float* dest, const float* source, int in_h, int in_w, int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradDataTransUnit8K3S1Pack8(dest, 8*8, source + w, ldh, ldc);
            dest += 8*8*8;
        }
        source += 6 * ldh;
    }
}

void winogradUVTransSliceUnit8K3S1Pack1(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradUVTransUnit8K3S1Pack1(dest + w, ldest, source_v, source_u, 8*1);
            source_v += 8*8*1;
        }
        dest += 6*ldest;
    }
}


void winogradUVTransSliceUnit8K3S1Pack4(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradUVTransUnit8K3S1Pack4(dest + w, ldest, source_v, source_u, 8*4);
            source_v += 8*8*4;
        }
        dest += 6*ldest;
    }
}

void winogradUVTransSliceUnit8K3S1Pack8(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc) {
    int h=0, w=0;
    for(h = 0; h < in_h - 7; h += 6) {
        for(w = 0; w < in_w - 7; w += 6) {
            kernel::winogradUVTransUnit8K3S1Pack8(dest + w, ldest, source_v, source_u, 8*8);
            source_v += 8*8*8;
        }
        dest += 6*ldest;
    }
}


}
}