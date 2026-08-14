cmake_minimum_required(VERSION 3.20)

# Primary target: sm_120 = consumer Blackwell (RTX 5090).
# Also: sm_100 = datacenter Blackwell (B200/GB200), sm_90 = Hopper, sm_89 = Ada.
# NOTE: RTX 5090 is compute capability 12.0 (sm_120), NOT sm_100. sm_100 is
# datacenter Blackwell (B200) and is binary-incompatible with the 5090.
# Must be set BEFORE project(... CUDA) — see root CMakeLists.txt for why.
# sm_121 excluded — unsupported by the CUDA 12.8 toolkit currently in use.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120" CACHE STRING "CUDA architectures")
project(sparkinfer_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_TESTS         "Build unit tests"            ON)
option(BUILD_BENCHMARKS    "Build benchmarks"            OFF)
# Experimental CuTe DSL kernels. Require CUTLASS and target sm_100a/sm_120a
# tensor-core paths. OFF by default; the portable CUDA path below is the
# production build and is what runs on RTX 5090 (sm_120).
option(BUILD_CUTE_KERNELS  "Experimental CuTe DSL kernels (needs CUTLASS)" OFF)
# Native SM120 block-scaled FP4 prefill kernels. Deliberately NOT an option(): option() writes a
# cache entry and then does nothing at all once one exists, and build directories here are reused
# across configures -- so a tree that was ever configured while these kernels were off would pin
# them off forever, and the default below would never be read again. A normal variable is
# recomputed every configure and shadows any stale cache entry of the same name.
# -DSPARKINFER_NVFP4=OFF opts out.
if(DEFINED SPARKINFER_NVFP4)
    set(BUILD_NVFP4_KERNELS ${SPARKINFER_NVFP4})
else()
    set(BUILD_NVFP4_KERNELS ON)
endif()

# CUTLASS is header-only and fetched at configure time. A box with no network must still be able
# to configure, so probe the remote once on a cold build tree and fall back to the portable CUDA
# path instead of failing the whole build.
if(BUILD_NVFP4_KERNELS AND
   NOT EXISTS "${CMAKE_BINARY_DIR}/_deps/cutlass-src/include/cutlass/cutlass.h")
    find_package(Git QUIET)
    if(NOT GIT_EXECUTABLE)
        set(BUILD_NVFP4_KERNELS OFF)
        message(STATUS "NVFP4: git not found -- building the portable CUDA prefill path")
    else()
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" ls-remote --exit-code
                    https://github.com/NVIDIA/cutlass.git v4.7.0
            RESULT_VARIABLE _nvfp4_reachable OUTPUT_QUIET ERROR_QUIET TIMEOUT 60)
        if(NOT _nvfp4_reachable EQUAL 0)
            set(BUILD_NVFP4_KERNELS OFF)
            message(STATUS "NVFP4: CUTLASS unreachable -- building the portable CUDA prefill path")
        endif()
    endif()
endif()

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
if(BUILD_CUTE_KERNELS OR BUILD_NVFP4_KERNELS)
    include(FetchContent)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        v4.7.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(cutlass)
endif()

if(BUILD_CUTE_KERNELS)
    add_subdirectory(csrc/cute/moe_swiglu)
    add_subdirectory(csrc/cute/moe_gemm)
    list(APPEND SPARKINFER_KERNEL_LIBS si_cute_moe_swiglu si_cute_moe_gemm)
endif()

if(BUILD_NVFP4_KERNELS)
    # A library of its OWN, compiled whole-program. The block-scaled FP4 MMA only assembles for
    # sm_120a, and a RELOCATABLE object at 120a cannot be device-linked against its 120 siblings
    # ("nvlink fatal: elfLink fatbinary error"). Turning separable compilation off keeps the device
    # link inside this one translation unit, so this is the only target that needs the arch and the
    # rest of the build keeps the architecture list it was configured with. The entry points are
    # host functions, and prefill_nvfp4_supported() gates on cc 12.0 at runtime, so a non-Blackwell
    # device simply never calls in.
    add_library(si_nvfp4 STATIC csrc/cuda/fused/prefill_nvfp4_sm120.cu)
    target_include_directories(si_nvfp4 PRIVATE include
        ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include)
    target_compile_options(si_nvfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(si_nvfp4 PROPERTIES
        CUDA_ARCHITECTURES "120a"
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
    # Compiles out the linkable fallbacks in prefill_nvfp4.cu, which si_fused owns.
    target_compile_definitions(si_fused PRIVATE SPARKINFER_BUILD_NVFP4=1)
    list(APPEND SPARKINFER_KERNEL_LIBS si_nvfp4)
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
