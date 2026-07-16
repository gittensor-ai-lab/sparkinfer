file(GLOB SRC "*.cu")
add_library(si_quant STATIC ${SRC})
target_include_directories(si_quant PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(si_quant PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--ptxas-options=-v>)
set_target_properties(si_quant PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
