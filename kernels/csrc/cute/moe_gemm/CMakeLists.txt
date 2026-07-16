file(GLOB_RECURSE SRC "*.cu" "*.cpp")
add_library(si_cute_moe_gemm STATIC ${SRC})
target_include_directories(si_cute_moe_gemm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/include
)
# CuTe DSL kernels require C++17 and extended lambdas
target_compile_options(si_cute_moe_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 -lineinfo
        --use_fast_math
        --extended-lambda
        --expt-relaxed-constexpr
        --ptxas-options=-v
    >
)
