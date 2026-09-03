cmake_minimum_required(VERSION 3.20)

# Primary target sm_120 (RTX 5090). Also Ada/Hopper/datacenter-Blackwell.
# Must be set BEFORE project(... CUDA) — see root CMakeLists.txt for why.
# sm_121 excluded — unsupported by the CUDA 12.8 toolkit currently in use.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120" CACHE STRING "CUDA architectures")
project(sparkinfer_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_TESTS    "Build unit / integration tests" ON)
option(BUILD_EXAMPLES "Build example programs"          ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)   # thermal governor's inter-token pace uses std::this_thread

# This repo is the integrator: pull in sparkinfer-kernels and sparkinfer-moe.
# In a superbuild the targets may already exist; standalone, add the siblings.
if(NOT TARGET sparkinfer_kernels)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kernels ${CMAKE_BINARY_DIR}/_kernels)
endif()
if(NOT TARGET sparkinfer_moe)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../moe ${CMAKE_BINARY_DIR}/_moe)
endif()

include_directories(include)

file(GLOB_RECURSE RUNTIME_SRC "src/*.cpp" "csrc/cuda/*.cu")
if(WIN32)
  add_library(sparkinfer_runtime STATIC ${RUNTIME_SRC})
else()
  add_library(sparkinfer_runtime SHARED ${RUNTIME_SRC})
endif()
target_include_directories(sparkinfer_runtime PUBLIC include)
# DFlashDraftModel::load_gguf (src/models/dflash_draft.cpp) derives its config from
# examples/dflash_gguf_config.h, the same header-only convention examples/qwen3_gguf_config.h
# already uses for the target model -- PRIVATE so it does not leak into the library's public
# include surface.
target_include_directories(sparkinfer_runtime PRIVATE examples)
target_link_libraries(sparkinfer_runtime PUBLIC sparkinfer_moe sparkinfer_kernels CUDA::cudart Threads::Threads)

# GPU observability (heat/power/clock) uses NVML when available; VRAM works without it. NVML ships
# with the CUDA toolkit as CUDA::nvml — link + flag it when present so the build degrades cleanly
# on boxes/toolkits without the NVML stub.
if(TARGET CUDA::nvml)
    target_link_libraries(sparkinfer_runtime PUBLIC CUDA::nvml)
    target_compile_definitions(sparkinfer_runtime PRIVATE SPARKINFER_HAVE_NVML)
    message(STATUS "sparkinfer: NVML found — GPU temperature/power observability enabled")
else()
    message(STATUS "sparkinfer: NVML not found — GPU observability limited to VRAM")
endif()
target_compile_options(sparkinfer_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>)
set_target_properties(sparkinfer_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_executable(qwen35_generate examples/qwen35_generate.cpp)
    target_include_directories(qwen35_generate PRIVATE include)
    target_link_libraries(qwen35_generate PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_generate examples/qwen3_gguf_generate.cpp)
    target_include_directories(qwen3_gguf_generate PRIVATE include)
    find_package(Threads REQUIRED)   # background GPU-observability sampler uses std::thread
    # nlohmann_json: same reason as qwen3_gguf_bench below -- this tool also routes through
    # qwen_checkpoint.h so a compressed-tensors directory can be generated from directly.
    target_link_libraries(qwen3_gguf_generate PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads
                          nlohmann_json::nlohmann_json)
    add_executable(qwen3_gguf_bench examples/qwen3_gguf_bench.cpp)
    target_include_directories(qwen3_gguf_bench PRIVATE include)
    # nlohmann_json: qwen_checkpoint.h reads an HF compressed-tensors (NVFP4) config.json via
    # qwen38_hf_config.h, so the PR eval bot can benchmark the SAME checkpoint the server serves.
    # Declared at the top level (see the root CMakeLists) because runtime/ is configured before
    # server/. Links into this executable only -- sparkinfer_runtime stays JSON-free.
    target_link_libraries(qwen3_gguf_bench PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)
    # TTFT benchmark: LMCache cache-hit vs. full-recompute baseline on a synthetic model (see
    # lmcache_bench.cpp's header comment for why a real GGUF isn't needed). Requires a real
    # sidecar venv at runtime (SPARKINFER_LMCACHE_PYTHON/_BRIDGE_SCRIPT), not build time.
    add_executable(lmcache_bench examples/lmcache_bench.cpp)
    target_include_directories(lmcache_bench PRIVATE include)
    target_link_libraries(lmcache_bench PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_cb_bench examples/qwen3_gguf_cb_bench.cpp)
    target_include_directories(qwen3_gguf_cb_bench PRIVATE include)
    find_package(Threads REQUIRED)
    target_link_libraries(qwen3_gguf_cb_bench PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
    add_executable(qwen3_gguf_score examples/qwen3_gguf_score.cpp)
    target_include_directories(qwen3_gguf_score PRIVATE include)
    target_link_libraries(qwen3_gguf_score PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)   # same reason as qwen3_gguf_bench above
    # Model-free NVFP4 GEMM correctness check: isolates the kernel from the loader, the KV cache
    # and 64 layers of accumulation, so a shape-dependent defect shows up on its own.
    add_executable(nvfp4_gemm_check examples/nvfp4_gemm_check.cpp)
    target_include_directories(nvfp4_gemm_check PRIVATE include)
    target_link_libraries(nvfp4_gemm_check PRIVATE sparkinfer_runtime CUDA::cudart)

    add_executable(qwen3_gguf_prefill_check examples/qwen3_gguf_prefill_check.cpp)
    target_include_directories(qwen3_gguf_prefill_check PRIVATE include)
    # nlohmann_json: same reason as qwen3_gguf_bench above -- this tool now routes through
    # qwen_checkpoint.h so it can A/B batched prefill on a compressed-tensors checkpoint, which is
    # the only place the FP4 FFN prefill arm is reachable.
    target_link_libraries(qwen3_gguf_prefill_check PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)
    add_executable(qwen3_gguf_dflash_check examples/qwen3_gguf_dflash_check.cpp)
    target_include_directories(qwen3_gguf_dflash_check PRIVATE include)
    target_link_libraries(qwen3_gguf_dflash_check PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_dflash_bench examples/qwen3_gguf_dflash_bench.cpp)
    target_include_directories(qwen3_gguf_dflash_bench PRIVATE include)
    target_link_libraries(qwen3_gguf_dflash_bench PRIVATE sparkinfer_runtime CUDA::cudart)
    # DSpark draft load check: exercises DFlashDraftModel::load() against the
    # RadixArk/Qwen3.8-27B-DSpark checkpoint and prints the parsed config.
    add_executable(dspark_load_check examples/dspark_load_check.cpp)
    target_include_directories(dspark_load_check PRIVATE include)
    target_link_libraries(dspark_load_check PRIVATE sparkinfer_runtime CUDA::cudart)
    # DSpark tau measurement: acceptance length of the DSpark draft against Qwen3.8-27B.
    add_executable(dspark_tau_check examples/dspark_tau_check.cpp)
    target_include_directories(dspark_tau_check PRIVATE include)
    target_link_libraries(dspark_tau_check PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)
    # Vision-tower probe: does this checkpoint carry a complete vision encoder, and does every
    # tensor match the shape its own vision_config implies? Stage 1 of image support -- no
    # inference, just a structural check, because a silently missing or mis-shaped vision weight
    # produces plausible garbage rather than an error.
    add_executable(vision_probe examples/vision_probe.cpp)
    target_include_directories(vision_probe PRIVATE include)
    target_link_libraries(vision_probe PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)
    # Runs the CUDA tower on vision_ref.py's exact synthetic input and dumps the embeddings for
    # that script to judge. The reference owns the tolerance, not this binary.
    # Cross-checks the smart_resize port against the Python original over a grid of shapes.
    # Exact integer logic, so bench/scripts/vision_resize_check.py demands strict equality.
    add_executable(vision_resize_check examples/vision_resize_check.cpp)
    target_include_directories(vision_resize_check PRIVATE include)
    target_link_libraries(vision_resize_check PRIVATE sparkinfer_runtime CUDA::cudart)
    # Splice behaviour, especially the REFUSALS -- a count mismatch must be a hard error, never
    # a partial write, or the model silently reads a truncated image region.
    add_executable(vision_splice_check examples/vision_splice_check.cpp)
    target_include_directories(vision_splice_check PRIVATE include)
    target_link_libraries(vision_splice_check PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(vision_forward_check examples/vision_forward_check.cpp)
    target_include_directories(vision_forward_check PRIVATE include)
    target_link_libraries(vision_forward_check PRIVATE sparkinfer_runtime CUDA::cudart
                          nlohmann_json::nlohmann_json)
    add_executable(thermal_sweep examples/thermal_sweep.cpp)   # force each thermal mode, measure W/°C/tok/s
    target_include_directories(thermal_sweep PRIVATE include)
    target_link_libraries(thermal_sweep PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
endif()
