cmake_minimum_required(VERSION 3.10)
project(proj)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )

FIND_PACKAGE(PkgConfig REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(/usr/local/cuda-9.0/include)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/gcc-5)
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)

# gui library gtkmm-3.0
pkg_check_modules(GTKMM gtkmm-3.0)
link_directories(${GTKMM_LIBRARY_DIRS})
include_directories(include ${GTKMM_INCLUDE_DIRS})


# jpg library libjpeg
pkg_check_modules(LIBJPG libjpeg)
link_directories(${LIBJPG_LIBRARY_DIRS})
include_directories(include ${LIBJPG_INCLUDE_DIRS})


## add executable
CUDA_ADD_EXECUTABLE(proj main.cu simulation.cpp simulation.h numerical_kernels.h)
set_property(TARGET proj PROPERTY CUDA_STANDARD 11)


set_target_properties( proj
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(proj ${GTKMM_LIBRARIES})
target_link_libraries(proj ${LIBJPG_LIBRARIES})
target_link_libraries(proj /usr/local/cuda/lib64/libcudart.so)

