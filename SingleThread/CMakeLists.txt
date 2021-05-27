cmake_minimum_required(VERSION 3.17)
project(lab12_1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab12_1 main.cu)

set_target_properties(
        lab12_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)