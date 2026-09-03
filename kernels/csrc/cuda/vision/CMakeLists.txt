# Vision-tower kernels: LayerNorm, tanh-GELU, full bidirectional patch attention, 2x2 merge.
# Separate from si_fused because none of it is shared with the text path -- the text model is
# RMSNorm + SwiGLU with causal/windowed attention throughout.
#
# NOTE: no --use_fast_math here, unlike the sibling domains. This tower is validated against
# bench/scripts/vision_ref.py, and fast-math's reassociation and lower-precision tanhf/expf would
# put a floor under that diff that has nothing to do with whether the port is correct. Speed is
# not the constraint: 27 blocks at hidden=1152 over a few thousand patches is a rounding error
# beside a 27B decode step.
file(GLOB SRC "*.cu")
add_library(si_vision STATIC ${SRC})
target_include_directories(si_vision PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(si_vision PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--ptxas-options=-v>)
set_target_properties(si_vision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
