cmake_minimum_required(VERSION 3.20)

# Blackwell consumer/edge: sm_120 (RTX 5090, RTX PRO 6000). sm_121 (RTX Spark, Jetson
# Thor) is intentionally excluded — CUDA 12.8's nvcc (currently in use) hard-errors with
# "Unsupported gpu architecture 'compute_121'" during compiler identification, since that
# target requires CUDA 12.9+.
# Must be set BEFORE project(... CUDA) — project()/enable_language(CUDA) reads
# CMAKE_CUDA_ARCHITECTURES at that call and caches its own (environment-dependent,
# sometimes ancient e.g. "52") default if it isn't already set. A set() after
# project() is a CACHE no-op once that's happened, silently building tensor-core/wmma
# kernels for a target that can't support them.
#
# The Blackwell entry is sm_120a, not sm_120, when the native FP4 prefill path is built. FP4's
# block-scaled MMA is an architecture-specific instruction: ptxas rejects
# `mma.sync.aligned.m16n8k64.kind::mxf4` outright on `.target sm_120` and accepts it on sm_120a,
# and sm_120a SASS runs on the same silicon. It has to be decided HERE, for the whole project, and
# not per target: a single library built for 120a while its siblings are 120 does not device-link
# at all ("nvlink fatal: elfLink fatbinary error"), which is why the FP4 kernels could not be
# enabled as they stood. The other architectures are unaffected.
include(cmake/Nvfp4Arch.cmake)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;${SPARKINFER_BLACKWELL_ARCH}"
    CACHE STRING "CUDA architectures")
project(sparkinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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
