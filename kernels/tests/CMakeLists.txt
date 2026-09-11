# CPU reference correctness tests — pure C++, no CUDA needed. They validate the
# kernel algorithms against an independent double-precision ground truth.
add_executable(cpu_reference_test cpu_reference_test.cpp)
set_target_properties(cpu_reference_test PROPERTIES CXX_STANDARD 17)
add_test(NAME cpu_reference_test COMMAND cpu_reference_test)

if(BUILD_NVFP4_KERNELS)
  add_executable(prefill_nvfp4_gpu_test prefill_nvfp4_gpu_test.cpp)
  # si_nvfp4, not si_fused: the FP4 entry points live in their own whole-program library so that
  # the sm_120a device code never has to device-link against the rest of the build.
  target_link_libraries(prefill_nvfp4_gpu_test PRIVATE si_nvfp4 si_fused si_quant CUDA::cudart)
  set_target_properties(prefill_nvfp4_gpu_test PROPERTIES CUDA_ARCHITECTURES "120a")
  add_test(NAME prefill_nvfp4_gpu_test COMMAND prefill_nvfp4_gpu_test)
  set_tests_properties(prefill_nvfp4_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The chunked row-batched down projection against the same rows run as their own batch. The rows
# kernel is instantiated for 2..8, so a wider decode batch is walked in chunks and every chunk past
# the first reads its inputs at an offset -- this pins that arithmetic.
add_executable(down_rows_chunk_gpu_test down_rows_chunk_gpu_test.cpp)
# sparkinfer_kernels, not the individual archives: it wraps them in --start-group so the
# single-pass linker scan cannot drop a member (see the note in kernels/CMakeLists.txt).
target_link_libraries(down_rows_chunk_gpu_test PRIVATE sparkinfer_kernels CUDA::cudart)
set_target_properties(down_rows_chunk_gpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME down_rows_chunk_gpu_test COMMAND down_rows_chunk_gpu_test)
set_tests_properties(down_rows_chunk_gpu_test PROPERTIES SKIP_RETURN_CODE 77)

# The row-batched contextual-sparsity gate/up against the per-token kernel it replaces: same rows,
# same bits, at every batch width the dispatch instantiates. Needs a GPU and ~350 MB of it for Muse
# Glimmer's real 6656x19968 FFN, which is the shape the arm is gated to.
add_executable(sparse_gate_up_rows_gpu_test sparse_gate_up_rows_gpu_test.cpp)
target_link_libraries(sparse_gate_up_rows_gpu_test PRIVATE sparkinfer_kernels CUDA::cudart)
set_target_properties(sparse_gate_up_rows_gpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME sparse_gate_up_rows_gpu_test COMMAND sparse_gate_up_rows_gpu_test)
set_tests_properties(sparse_gate_up_rows_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
