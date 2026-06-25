add_executable(engine_guard_gpu_test engine_guard_gpu_test.cpp)
target_link_libraries(engine_guard_gpu_test PRIVATE sparkinfer_moe CUDA::cudart)
add_test(NAME engine_guard_gpu_test COMMAND engine_guard_gpu_test)
