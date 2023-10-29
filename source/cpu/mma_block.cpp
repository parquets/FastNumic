#include "mma_block.hpp"

namespace fastnum {
namespace cpu {

void mma_block(int mc, int nc, int kc, const float *packA, const float *packB, float *C, int ldc) {
    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;

    //printf("%d %d %d\n", mc, nc, kc);
    // printf("%.2f %.2f %.2f %.2f\n", packA[0], packA[1], packA[2], packA[3]);

    for (mm = 0; mm < mc - 5; mm += 6) {
        offset_b = 0;
        for (nn = 0; nn < nc - 15; nn += 16) {
            kernel::mma_6x16(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 16 * kc;
        }
        for (; nn < nc - 7; nn += 8) {
            kernel::mma_6x8(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        for (; nn < nc - 3; nn += 4) {
            kernel::mma_6x4(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 4 * kc;
        }
        for (; nn < nc; ++nn) {
            kernel::mma_6x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += 6 * kc;
    }
    for (; mm < mc - 3; mm += 4) {
        offset_b = 0;

        for (nn = 0; nn < nc - 15; nn += 16) {
            kernel::mma_4x16(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 16 * kc;
        }
        for (; nn < nc - 7; nn += 8) {
            kernel::mma_4x8(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        for (; nn < nc - 3; nn += 4) {
            kernel::mma_4x4(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 4 * kc;
        }
        for (; nn < nc; ++nn) {
            kernel::mma_4x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += 4 * kc;
    }
    for (; mm < mc; ++mm) {
        offset_b = 0;
        for (nn = 0; nn < nc - 15; nn += 16) {
            kernel::mma_1x16(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 16 * kc;
        }
        for (; nn < nc - 7; nn += 8) {
            kernel::mma_1x8(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        for (; nn < nc - 3; nn += 4) {
            kernel::mma_1x4(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 4 * kc;
        }
        for (; nn < nc; ++nn) {
            kernel::mma_1x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += kc;
    }
}


}  // namespace cpu
}  // namespace fastnum