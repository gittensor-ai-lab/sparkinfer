cmake_minimum_required(VERSION 3.20)
project(sparkinfer_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Primary target sm_120 (RTX 5090). Also Ada/Hopper/datacenter-Blackwell.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120;121" CACHE STRING "CUDA architectures")

option(BUILD_TESTS    "Build unit / integration tests" ON)
option(BUILD_EXAMPLES "Build example programs"          ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)   # thermal governor's inter-token pace uses std::this_thread

# This repo is the integrator: pull in sparkinfer-kernels and sparkinfer-moe.
# In a superbuild the targets may already exist; standalone, add the siblings.
if(NOT TARGET sparkinfer_kernels)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kernels ${CMAKE_BINARY_DIR}/_kernels)
endif()
if(NOT TARGET sparkinfer_moe)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../moe ${CMAKE_BINARY_DIR}/_moe)
endif()

include_directories(include)

file(GLOB_RECURSE RUNTIME_SRC "src/*.cpp" "csrc/cuda/*.cu")
add_library(sparkinfer_runtime SHARED ${RUNTIME_SRC})
target_include_directories(sparkinfer_runtime PUBLIC include)
target_link_libraries(sparkinfer_runtime PUBLIC sparkinfer_moe sparkinfer_kernels CUDA::cudart Threads::Threads)

# GPU observability (heat/power/clock) uses NVML when available; VRAM works without it. NVML ships
# with the CUDA toolkit as CUDA::nvml — link + flag it when present so the build degrades cleanly
# on boxes/toolkits without the NVML stub.
if(TARGET CUDA::nvml)
    target_link_libraries(sparkinfer_runtime PUBLIC CUDA::nvml)
    target_compile_definitions(sparkinfer_runtime PRIVATE SPARKINFER_HAVE_NVML)
    message(STATUS "sparkinfer: NVML found — GPU temperature/power observability enabled")
else()
    message(STATUS "sparkinfer: NVML not found — GPU observability limited to VRAM")
endif()
target_compile_options(sparkinfer_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>)
set_target_properties(sparkinfer_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_executable(qwen35_generate examples/qwen35_generate.cpp)
    target_include_directories(qwen35_generate PRIVATE include)
    target_link_libraries(qwen35_generate PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_generate examples/qwen3_gguf_generate.cpp)
    target_include_directories(qwen3_gguf_generate PRIVATE include)
    find_package(Threads REQUIRED)   # background GPU-observability sampler uses std::thread
    target_link_libraries(qwen3_gguf_generate PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
    add_executable(qwen3_gguf_bench examples/qwen3_gguf_bench.cpp)
    target_include_directories(qwen3_gguf_bench PRIVATE include)
    target_link_libraries(qwen3_gguf_bench PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_score examples/qwen3_gguf_score.cpp)
    target_include_directories(qwen3_gguf_score PRIVATE include)
    target_link_libraries(qwen3_gguf_score PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_prefill_check examples/qwen3_gguf_prefill_check.cpp)
    target_include_directories(qwen3_gguf_prefill_check PRIVATE include)
    target_link_libraries(qwen3_gguf_prefill_check PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(thermal_sweep examples/thermal_sweep.cpp)   # force each thermal mode, measure W/°C/tok/s
    target_include_directories(thermal_sweep PRIVATE include)
    target_link_libraries(thermal_sweep PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
endif()
