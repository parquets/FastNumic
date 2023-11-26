#include "sgemm.cuh"

namespace fastnum {
namespace cuda {
namespace kernel {


__global__ void sgemm_nn(int M, int N, int K, 
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {
    
} 

} // namespace kernel;
} // namespace cuda
} // namespace fastnum