cmake_minimum_required(VERSION 3.12)

add_subdirectory(cpu)

if(BUILD_WITH_CUDA)
    add_subdirectory(cuda)
endif()

set(FastNumSource
    gemm.cpp
    gemv.cpp
    transpose.cpp
    # convolution.cpp
    convolution1d.cpp
    im2col.cpp
    arithemetic.cpp
)

add_library(FastNum STATIC ${FastNumSource})

target_include_directories(FastNum PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_link_libraries(FastNum PUBLIC FastNumCPU)

if(BUILD_WITH_CUDA)
    target_link_libraries(FastNum PUBLIC FastNumCUDA ${CUDA_LIBRARIES})
endif()