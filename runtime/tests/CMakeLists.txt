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

add_executable(temperature_sample_gpu_test temperature_sample_gpu_test.cpp)
target_link_libraries(temperature_sample_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME temperature_sample_gpu_test COMMAND temperature_sample_gpu_test)

add_executable(topk_topp_sample_gpu_test topk_topp_sample_gpu_test.cpp)
target_link_libraries(topk_topp_sample_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME topk_topp_sample_gpu_test COMMAND topk_topp_sample_gpu_test)

add_executable(logprob_extract_gpu_test logprob_extract_gpu_test.cpp)
target_link_libraries(logprob_extract_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME logprob_extract_gpu_test COMMAND logprob_extract_gpu_test)

add_executable(penalty_gpu_test penalty_gpu_test.cpp)
target_link_libraries(penalty_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME penalty_gpu_test COMMAND penalty_gpu_test)

add_executable(logit_bias_gpu_test logit_bias_gpu_test.cpp)
target_link_libraries(logit_bias_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME logit_bias_gpu_test COMMAND logit_bias_gpu_test)

# Real sidecar subprocess + real KV cache + a small synthetic model on real hardware -- the one
# place both halves of the LMCache bridge run together against the real model code path. Skips
# (not fails) unless SPARKINFER_LMCACHE_PYTHON/_BRIDGE_SCRIPT point at a real lmcache+torch venv
# (see bridge/README.md) -- most environments running the rest of this suite won't have that
# ~1.5GB, CPU-only-torch-is-enough dependency installed.
add_executable(lmcache_e2e_gpu_test lmcache_e2e_gpu_test.cpp)
target_link_libraries(lmcache_e2e_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME lmcache_e2e_gpu_test COMMAND lmcache_e2e_gpu_test)

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

# Video preprocessing — pure C++, no GPU, no checkpoint. Anchors the video path to the image
# path by EQUIVALENCE: a 2-frame video of one still must be byte-identical to that still as an
# image, because the tower's Conv3d time extent is filled by repetition for images and by real
# frames for video. That single assertion transfers the image path's measured reference-exactness
# (PIL bicubic constant, the twice-rounded uint8 intermediate, the (C,T,P,P) layout) onto video
# without re-verifying any of it against transformers.
add_executable(qwen_vision_video_cpu_test qwen_vision_video_cpu_test.cpp
               ../src/models/qwen_vision_preprocess.cpp)
target_include_directories(qwen_vision_video_cpu_test PRIVATE ../include)
set_target_properties(qwen_vision_video_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen_vision_video_cpu_test COMMAND qwen_vision_video_cpu_test)
