cmake_minimum_required(VERSION 3.12)

find_package(CUDA REQUIRED)

enable_language(CUDA)

message(STATUS "cuda version:" ${CUDA_VERSION_STRING})
message(STATUS ${CUDA_INCLUDE_DIRS})

set(FastNumCUDASource
    transpose.cu
)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(FastNumCUDA STATIC ${FastNumCUDASource})

target_link_libraries(FastNumCUDA PRIVATE ${CUDA_LIBRARIES})
target_include_directories(FastNumCUDA PRIVATE ${CUDA_INCLUDE_DIRS})