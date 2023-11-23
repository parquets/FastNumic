cmake_minimum_required(VERSION 3.12)

find_package(CUDA REQUIRED)

add_executable(transpose_cuda transpose_cuda.cpp)

target_link_libraries(transpose_cuda PRIVATE FastNum ${CUDA_LIBRARIES})

target_include_directories(transpose_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum ${CUDA_INCLUDE_DIRS})