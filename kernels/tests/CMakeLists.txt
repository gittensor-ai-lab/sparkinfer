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
# The tensor-core attention projection against the MMVQ, dumped per process for a cross-run diff.
add_executable(mmvq_mma_accuracy_gpu_test mmvq_mma_accuracy_gpu_test.cpp)
target_link_libraries(mmvq_mma_accuracy_gpu_test PRIVATE sparkinfer_kernels CUDA::cudart)
set_target_properties(mmvq_mma_accuracy_gpu_test PROPERTIES CXX_STANDARD 17)
# A request larger than the MMA row limit must still let its eight-row chunks use MMA. K=8192
# is supported by MMA but not the exact DP4A dispatcher, so bypassing MMA rejects this shape.
add_test(NAME mmvq_mma_chunk_gpu_test COMMAND mmvq_mma_accuracy_gpu_test 40 6656 8192)
set_tests_properties(mmvq_mma_chunk_gpu_test PROPERTIES SKIP_RETURN_CODE 77
  ENVIRONMENT "SPARKINFER_MMVQ_MMA=1;SPARKINFER_MMVQ_MMA_MINM=8;SPARKINFER_MMVQ_MMA_MINN=2048")
# The tensor-core down projection against the MMVQ, dumped per process so the two can be diffed.
# Not a ctest: it needs two runs with different env to be meaningful, and ~1 GB of device memory.
add_executable(down_mma_accuracy_gpu_test down_mma_accuracy_gpu_test.cpp)
target_link_libraries(down_mma_accuracy_gpu_test PRIVATE sparkinfer_kernels CUDA::cudart)
set_target_properties(down_mma_accuracy_gpu_test PROPERTIES CXX_STANDARD 17)
# The tensor-core LM head against the dp4a multi-row kernel it replaces. Both arms are exported
# functions, so one process covers both. Not a ctest: it needs ~0.8 GB of device memory.
add_executable(head_mma_accuracy_gpu_test head_mma_accuracy_gpu_test.cpp)
target_link_libraries(head_mma_accuracy_gpu_test PRIVATE sparkinfer_kernels CUDA::cudart)
set_target_properties(head_mma_accuracy_gpu_test PROPERTIES CXX_STANDARD 17)
