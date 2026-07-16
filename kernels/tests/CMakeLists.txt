# CPU reference correctness tests — pure C++, no CUDA needed. They validate the
# kernel algorithms against an independent double-precision ground truth.
add_executable(cpu_reference_test cpu_reference_test.cpp)
set_target_properties(cpu_reference_test PROPERTIES CXX_STANDARD 17)
add_test(NAME cpu_reference_test COMMAND cpu_reference_test)
