#include "fastnum/batch_transpose.hpp"
#include "fastnum/transpose.hpp"

namespace fastnum {
namespace cpu {

void batch_transpose(int batch, int M, int N, const float* A, int ldbatch_a, int lda, float* B, int ldbatch_b, int ldb) {
    for (int b = 0; b < batch; ++b) {
        transpose(M, N, A, lda, B, ldb);
        A += ldbatch_a;
        B += ldbatch_b;
    }
}

}  // namespace cpu
}  // namespace fastnum