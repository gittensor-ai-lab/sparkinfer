cmake_minimum_required(VERSION 3.20)
project(sparkinfer LANGUAGES CXX CUDA)

# Blackwell consumer/edge: sm_120 (RTX 5090, RTX PRO 6000), sm_121 (RTX Spark, Jetson Thor).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120;121" CACHE STRING "CUDA architectures")

option(BUILD_TESTS    "Build unit / integration tests" ON)
option(BUILD_EXAMPLES "Build example programs"          ON)

find_package(CUDAToolkit REQUIRED)

enable_testing()   # so `ctest` at the build root discovers the subdirs' add_test() registrations

# Superbuild. moe/ and runtime/ guard their sibling add_subdirectory(../kernels)
# with if(NOT TARGET ...), so adding kernels first here makes them reuse the
# already-defined targets instead of re-adding. Dependency order: kernels -> moe -> runtime.
add_subdirectory(kernels)
add_subdirectory(moe)
add_subdirectory(runtime)
# bench/ is benchmark scripts + frozen configs + results — not a CMake target.

option(BUILD_SERVER "Build sparkinfer-server (OpenAI-compatible HTTP API)" OFF)
if(BUILD_SERVER)
  add_subdirectory(server)
endif()
