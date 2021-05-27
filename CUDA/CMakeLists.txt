cmake_minimum_required(VERSION 3.17)
project(lab12_3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab12_3 main.cu)

set_target_properties(
        lab12_3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)