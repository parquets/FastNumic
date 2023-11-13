#include "pack_block.hpp"

namespace fastnum {
namespace cpu {

void spackA_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack) {
    int x = 0;
    for (x = 0; x < xc - 5; x += 6) {
        kernel::spack_h6(packed_data, X, ldx, n_pack);
        packed_data += (6 * yc);
        X += (6 * ldx);
    }
    for (; x < xc - 3; x += 4) {
        kernel::spack_h4(packed_data, X, ldx, n_pack);
        packed_data += (4 * yc);
        X += (4 * ldx);
    }
    for (; x < xc; ++x) {
        kernel::spack_h1(packed_data, X, ldx, n_pack);
        packed_data += (1 * yc);
        X += (1 * ldx);
    }
}

void spackA_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack) {
    int y = 0;
    for (y = 0; y < yc - 5; y += 6) {
        kernel::spack_v6(packed_data, X, ldx, n_pack);
        packed_data += (6 * xc);
        X += (6);
    }
    for (; y < yc - 3; y += 4) {
        kernel::spack_v4(packed_data, X, ldx, n_pack);
        packed_data += (4 * xc);
        X += (4);
    }
    for (; y < yc; ++y) {
        kernel::spack_v1(packed_data, X, ldx, n_pack);
        packed_data += (1 * xc);
        X += (1);
    }
}

void spackB_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack) {
    int x = 0;
    for (x = 0; x < xc - 15; x += 16) {
        kernel::spack_h16(packed_data, X, ldx, n_pack);
        packed_data += (16 * yc);
        X += (16 * ldx);
    }
    for (; x < xc - 7; x += 8) {
        kernel::spack_h8(packed_data, X, ldx, n_pack);
        packed_data += (8 * yc);
        X += (8 * ldx);
    }
    for (; x < xc - 3; x += 4) {
        kernel::spack_h4(packed_data, X, ldx, n_pack);
        packed_data += (4 * yc);
        X += (4 * ldx);
    }
    for (; x < xc; ++x) {
        kernel::spack_h1(packed_data, X, ldx, n_pack);
        packed_data += (1 * yc);
        X += (1 * ldx);
    }
}

void spackB_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack) {
    int y = 0;
    for (y = 0; y < yc - 15; y += 16) {
        kernel::spack_v16(packed_data, X, ldx, n_pack);
        packed_data += (16 * xc);
        X += (16);
    }
    for (; y < yc - 7; y += 8) {
        kernel::spack_v8(packed_data, X, ldx, n_pack);
        packed_data += (8 * xc);
        X += (8);
    }
    for (; y < yc - 3; y += 4) {
        kernel::spack_v4(packed_data, X, ldx, n_pack);
        packed_data += (4 * xc);
        X += (4);
    }
    for (; y < yc; ++y) {
        kernel::spack_v1(packed_data, X, ldx, n_pack);
        packed_data += (1 * xc);
        X += (1);
    }
}

void dpackA_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack) {
    int x = 0;
    for (; x < xc - 3; x += 4) {
        kernel::dpack_h4(packed_data, X, ldx, n_pack);
        packed_data += (4 * yc);
        X += (4 * ldx);
    }
    for (; x < xc; ++x) {
        kernel::dpack_h1(packed_data, X, ldx, n_pack);
        packed_data += (1 * yc);
        X += (1 * ldx);
    }
}

void dpackA_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack) {
    int y = 0;
    // printf("%d %d %d\n", xc, yc, n_pack);
    // printf("before pack\n");
    // for(int ix=0; ix<xc; ++ix) {
    //     for(int iy=0; iy<yc; ++iy) {
    //         printf("%.5f ", X[ix*ldx+iy]);
    //     }
    //     printf("\n");
    // }
    for (; y < yc - 3; y += 4) {
        kernel::dpack_v4(packed_data, X, ldx, n_pack);
        packed_data += (4 * xc);
        X += (4);
    }
    for (; y < yc; ++y) {
        kernel::dpack_v1(packed_data, X, ldx, n_pack);
        packed_data += (1 * xc);
        X += (1);
    }
    // printf("after pack\n");
    // for(int i=0; i<xc*yc; ++i) {
    //     printf("%.5f ", packed_data[i]);
    // }

    //exit(0);
}

void dpackB_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack) {
    int x = 0;
    for (; x < xc - 7; x += 8) {
        kernel::dpack_h8(packed_data, X, ldx, n_pack);
        packed_data += (8 * yc);
        X += (8 * ldx);
    }
    for (; x < xc - 3; x += 4) {
        kernel::dpack_h4(packed_data, X, ldx, n_pack);
        packed_data += (4 * yc);
        X += (4 * ldx);
    }
    for (; x < xc; ++x) {
        kernel::dpack_h1(packed_data, X, ldx, n_pack);
        packed_data += (1 * yc);
        X += (1 * ldx);
    }
}

void dpackB_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack) {
    int y = 0;
    for (; y < yc - 7; y += 8) {
        kernel::dpack_v8(packed_data, X, ldx, n_pack);
        packed_data += (8 * xc);
        X += (8);
    }
    for (; y < yc - 3; y += 4) {
        kernel::dpack_v4(packed_data, X, ldx, n_pack);
        packed_data += (4 * xc);
        X += (4);
    }
    for (; y < yc; ++y) {
        kernel::dpack_v1(packed_data, X, ldx, n_pack);
        packed_data += (1 * xc);
        X += (1);
    }
}

}  // namespace cpu
}  // namespace fastnum