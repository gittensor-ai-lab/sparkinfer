cmake_minimum_required(VERSION 3.20)
project(sparkinfer_moe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120;121" CACHE STRING "CUDA architectures")

option(BUILD_TESTS "Build unit tests" ON)

find_package(CUDAToolkit REQUIRED)

# Depend on sparkinfer-kernels. In the superbuild the target already exists;
# standalone, pull in the sibling checkout.
if(NOT TARGET sparkinfer_kernels)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kernels ${CMAKE_BINARY_DIR}/_kernels)
endif()

include_directories(include)

file(GLOB_RECURSE MOE_SRC "src/*.cpp")
if(WIN32)
  add_library(sparkinfer_moe STATIC ${MOE_SRC})
else()
  add_library(sparkinfer_moe SHARED ${MOE_SRC})
endif()
target_include_directories(sparkinfer_moe PUBLIC include)
target_link_libraries(sparkinfer_moe PUBLIC sparkinfer_kernels CUDA::cudart)

if(BUILD_TESTS)
    enable_testing()
endif()
