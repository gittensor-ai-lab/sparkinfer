# CPU composition tests — pure C++, validate the wiring math (no GPU needed).
add_executable(decode_layer_cpu_test decode_layer_cpu_test.cpp)
set_target_properties(decode_layer_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME decode_layer_cpu_test COMMAND decode_layer_cpu_test)

add_executable(qwen35_cpu_test qwen35_cpu_test.cpp)
set_target_properties(qwen35_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen35_cpu_test COMMAND qwen35_cpu_test)

add_executable(qwen36_config_cpu_test qwen36_config_cpu_test.cpp ../src/gguf.cpp)
set_target_properties(qwen36_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen36_config_cpu_test COMMAND qwen36_config_cpu_test)

# Muse Glimmer DFlash draft GGUF config parsing. Links only gguf.cpp, same as
# qwen36_config_cpu_test above -- no kernels/moe/full runtime needed. dflash_draft.h (pulled in
# via ../examples/dflash_gguf_config.h for DFlashDraftConfig) needs <cuda_runtime.h> for
# cudaStream_t in DFlashDraftModel's method signatures even though this test never constructs
# one, so link CUDA::cudart for its headers -- lighter than sparkinfer_runtime, no device
# execution, still GPU-free in the sense that matters (no kernels, no MoE engine).
add_executable(dflash_gguf_config_cpu_test dflash_gguf_config_cpu_test.cpp ../src/gguf.cpp)
target_link_libraries(dflash_gguf_config_cpu_test PRIVATE CUDA::cudart)
set_target_properties(dflash_gguf_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME dflash_gguf_config_cpu_test COMMAND dflash_gguf_config_cpu_test)

# Thermal governor — pure tiering policy (no GPU); links the runtime for classify()/pace() symbols.
add_executable(thermal_governor_cpu_test thermal_governor_cpu_test.cpp)
target_link_libraries(thermal_governor_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(thermal_governor_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME thermal_governor_cpu_test COMMAND thermal_governor_cpu_test)

add_executable(scheduler_cpu_test scheduler_cpu_test.cpp)
target_link_libraries(scheduler_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(scheduler_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME scheduler_cpu_test COMMAND scheduler_cpu_test)

# BridgeClient (docs/lmcache_bridge_protocol.md) against a from-scratch fake AF_UNIX peer --
# no GPU, no real lmcache/Python, no shm KV staging (that's bridge/tests/test_bridge_e2e.py's
# job). Only needs lmcache_bridge_client.cpp, not the full sparkinfer_runtime -- it has zero CUDA
# dependencies (confirmed: compiles standalone with plain g++, no CUDA headers needed).
add_executable(lmcache_bridge_client_cpu_test lmcache_bridge_client_cpu_test.cpp
               ../src/lmcache_bridge_client.cpp)
target_link_libraries(lmcache_bridge_client_cpu_test PRIVATE Threads::Threads)
set_target_properties(lmcache_bridge_client_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME lmcache_bridge_client_cpu_test COMMAND lmcache_bridge_client_cpu_test)

add_executable(batch_engine_gpu_test batch_engine_gpu_test.cpp)
target_link_libraries(batch_engine_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME batch_engine_gpu_test COMMAND batch_engine_gpu_test)

# GPU integration tests — run the real device path; skip without a device.
add_executable(decode_runner_gpu_test decode_runner_gpu_test.cpp)
target_link_libraries(decode_runner_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME decode_runner_gpu_test COMMAND decode_runner_gpu_test)

add_executable(qwen35_gpu_test qwen35_gpu_test.cpp)
target_link_libraries(qwen35_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME qwen35_gpu_test COMMAND qwen35_gpu_test)

add_executable(dflash_gdn_checkpoint_gpu_test dflash_gdn_checkpoint_gpu_test.cpp)
target_link_libraries(dflash_gdn_checkpoint_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME dflash_gdn_checkpoint_gpu_test COMMAND dflash_gdn_checkpoint_gpu_test)
