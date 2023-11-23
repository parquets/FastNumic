cmake_minimum_required(VERSION 3.12)

project(FastNum)


option(BUILD_DEMO "Build demo/exec or not" ON)
option(BUILD_WITH_CUDA "Build tools/cpp or not" ON)


add_subdirectory(${PROJECT_SOURCE_DIR}/source)
add_subdirectory(${PROJECT_SOURCE_DIR}/samples)