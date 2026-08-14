# Decides whether the native SM120 block-scaled FP4 prefill kernels can be built, and therefore
# which Blackwell architecture the project targets.
#
# Included from the top-level CMakeLists BEFORE project(), and again from kernels/ (which is also
# configurable standalone). The arch has to be settled before project() because
# enable_language(CUDA) latches CMAKE_CUDA_ARCHITECTURES at that call.
include_guard(GLOBAL)

option(BUILD_NVFP4_KERNELS
       "SM120 native block-scaled FP4 dense prefill kernels (header-only CUTLASS)" ON)

# The FP4 kernels compile against header-only CUTLASS, fetched at configure time. A box without
# network access must still be able to configure the project, so probe the remote once and fall
# back to the portable CUDA path rather than hard-failing the whole build. A tree that has already
# fetched CUTLASS skips the probe, so this costs one `ls-remote` on a cold build directory.
if(BUILD_NVFP4_KERNELS)
    set(_nvfp4_deps "${FETCHCONTENT_BASE_DIR}")
    if(NOT _nvfp4_deps)
        set(_nvfp4_deps "${CMAKE_BINARY_DIR}/_deps")
    endif()
    if(NOT EXISTS "${_nvfp4_deps}/cutlass-src/include/cutlass/cutlass.h")
        find_package(Git QUIET)
        if(NOT GIT_EXECUTABLE)
            set(BUILD_NVFP4_KERNELS OFF CACHE BOOL "" FORCE)
            message(STATUS "NVFP4: git not found -- building the portable CUDA prefill path")
        else()
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" ls-remote --exit-code
                        https://github.com/NVIDIA/cutlass.git v4.7.0
                RESULT_VARIABLE _nvfp4_reachable
                OUTPUT_QUIET ERROR_QUIET TIMEOUT 60)
            if(NOT _nvfp4_reachable EQUAL 0)
                set(BUILD_NVFP4_KERNELS OFF CACHE BOOL "" FORCE)
                message(STATUS "NVFP4: CUTLASS unreachable -- building the portable CUDA prefill path")
            endif()
        endif()
    endif()
endif()

# sm_120a, not sm_120: ptxas rejects the block-scaled FP4 MMA on `.target sm_120` and accepts it on
# sm_120a, which runs on the same silicon. Every target must agree -- one library at 120a among
# siblings at 120 fails to device-link ("nvlink fatal: elfLink fatbinary error").
if(BUILD_NVFP4_KERNELS)
    set(SPARKINFER_BLACKWELL_ARCH "120a")
else()
    set(SPARKINFER_BLACKWELL_ARCH "120")
endif()
