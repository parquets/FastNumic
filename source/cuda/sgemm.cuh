#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace fastnum {
namespace cuda {
    
__host__ void sgemm_nn_cuda_wrap(int M, int N, int K, 
                        float alpha, 
                        const float* A, int lda, 
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc);

__host__ void sgemm_nt_cuda_wrap(int M, int N, int K, 
                            float alpha, 
                            const float* A, int lda, 
                            const float* B, int ldb,
                            float beta,
                            float* C, int ldc);

__host__ void sgemm_tn_cuda_wrap(int M, int N, int K, 
                                float alpha, 
                                const float* A, int lda, 
                                const float* B, int ldb,
                                float beta,
                                float* C, int ldc);

__host__ void sgemm_tt_cuda_wrap(int M, int N, int K, 
                                float alpha, 
                                const float* A, int lda, 
                                const float* B, int ldb,
                                float beta,
                                float* C, int ldc);
}
}