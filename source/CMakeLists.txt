cmake_minimum_required(VERSION 3.12)

add_subdirectory(cpu)

set(FastNumSource
    gemm.cpp
    gemv.cpp
)

add_library(FastNum STATIC ${FastNumSource})

target_include_directories(FastNum PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_link_libraries(FastNum PUBLIC FastNumCPU)