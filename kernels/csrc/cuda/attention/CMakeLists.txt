file(GLOB SRC "*.cu")
add_library(si_attn STATIC ${SRC})
target_include_directories(si_attn PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(si_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--ptxas-options=-v>)
set_target_properties(si_attn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
