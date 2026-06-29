# CPU composition tests — pure C++, validate the wiring math (no GPU needed).
add_executable(decode_layer_cpu_test decode_layer_cpu_test.cpp)
set_target_properties(decode_layer_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME decode_layer_cpu_test COMMAND decode_layer_cpu_test)

add_executable(qwen35_cpu_test qwen35_cpu_test.cpp)
set_target_properties(qwen35_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen35_cpu_test COMMAND qwen35_cpu_test)

# Links the real (CUDA-free) spec_decode.cpp directly -- not a reimplementation,
# so this exercises the exact draft/accept logic forward_tokens_batch() uses.
add_executable(spec_decode_cpu_test spec_decode_cpu_test.cpp ../src/spec_decode.cpp)
target_include_directories(spec_decode_cpu_test PRIVATE ../include)
set_target_properties(spec_decode_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME spec_decode_cpu_test COMMAND spec_decode_cpu_test)

# Thermal governor — pure tiering policy (no GPU); links the runtime for classify()/pace() symbols.
add_executable(thermal_governor_cpu_test thermal_governor_cpu_test.cpp)
target_link_libraries(thermal_governor_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(thermal_governor_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME thermal_governor_cpu_test COMMAND thermal_governor_cpu_test)

# GPU integration tests — run the real device path; skip without a device.
add_executable(decode_runner_gpu_test decode_runner_gpu_test.cpp)
target_link_libraries(decode_runner_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME decode_runner_gpu_test COMMAND decode_runner_gpu_test)

add_executable(qwen35_gpu_test qwen35_gpu_test.cpp)
target_link_libraries(qwen35_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME qwen35_gpu_test COMMAND qwen35_gpu_test)
