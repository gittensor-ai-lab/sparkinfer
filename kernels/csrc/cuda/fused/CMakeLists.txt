file(GLOB SRC "*.cu")
list(FILTER SRC EXCLUDE REGEX "(prefill_nvfp4|prefill_ptq1_fp4|packed_ptq1_fp4)_sm120[.]cu")
add_library(si_fused STATIC ${SRC})
target_include_directories(si_fused PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(si_fused PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--ptxas-options=-v>)
set_target_properties(si_fused PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
