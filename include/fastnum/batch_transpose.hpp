#pragma once

namespace fastnum {
namespace cpu {

void batch_transpose(int batch, int M, int N, const float* A, int ldbatch_a, int lda, float* B, int ldbatch_b, int ld);

}
}