cmake_minimum_required(VERSION 3.20)
project(sparkinfer_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Primary target: sm_120 = consumer Blackwell (RTX 5090).
# Also: sm_100 = datacenter Blackwell (B200/GB200), sm_90 = Hopper, sm_89 = Ada.
# NOTE: RTX 5090 is compute capability 12.0 (sm_120), NOT sm_100. sm_100 is
# datacenter Blackwell (B200) and is binary-incompatible with the 5090.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120;121" CACHE STRING "CUDA architectures")

option(BUILD_TESTS         "Build unit tests"            ON)
option(BUILD_BENCHMARKS    "Build benchmarks"            OFF)
# Experimental CuTe DSL kernels. Require CUTLASS and target sm_100a/sm_120a
# tensor-core paths. OFF by default; the portable CUDA path below is the
# production build and is what runs on RTX 5090 (sm_120).
option(BUILD_CUTE_KERNELS  "Experimental CuTe DSL kernels (needs CUTLASS)" OFF)

include_directories(include)

# --- Portable CUDA kernel domains (production path, runs on sm_120) ---
add_subdirectory(csrc/cuda/attention)   # flash decode variants (GQA, hd256, hd512)
add_subdirectory(csrc/cuda/gemm)        # GEMM, GEMV, batched GEMM
add_subdirectory(csrc/cuda/moe)         # MoE router, grouped GEMM + SwiGLU
add_subdirectory(csrc/cuda/quant)       # quantization helpers
add_subdirectory(csrc/cuda/fused)       # fused GEMM + epilogue (non-CuTe path)

set(SPARKINFER_KERNEL_LIBS si_attn si_gemm si_moe si_quant si_fused)

if(MSVC)
  # MSVC device link of RDC static libs into a DLL is brittle; whole-TU compile is fine.
  foreach(lib IN LISTS SPARKINFER_KERNEL_LIBS)
    set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  endforeach()
endif()

# --- Optional CuTe DSL kernel domains (experimental, sm_100a/sm_120a) ---
if(BUILD_CUTE_KERNELS)
    include(FetchContent)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(cutlass)
    add_subdirectory(csrc/cute/moe_swiglu)
    add_subdirectory(csrc/cute/moe_gemm)
    list(APPEND SPARKINFER_KERNEL_LIBS si_cute_moe_swiglu si_cute_moe_gemm)
endif()

# Unified interface library
add_library(sparkinfer_kernels INTERFACE)
target_link_libraries(sparkinfer_kernels INTERFACE ${SPARKINFER_KERNEL_LIBS})
target_include_directories(sparkinfer_kernels INTERFACE include)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
