cmake_minimum_required(VERSION 3.20)

include(FetchContent)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        c586c52f93f7b060753bd2388eb96a105cb7374d
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp)

# nlohmann_json is declared at the TOP-LEVEL CMakeLists (runtime/ needs it too and is configured
# first); this target just uses it.

# JSON Schema patterns come from untrusted API requests. Use RE2's bounded,
# linear-time engine instead of std::regex, whose backtracking implementation
# can be driven into exponential runtime by a short adversarial pattern/value.
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(RE2_USE_ICU OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    re2
    URL      https://github.com/google/re2/archive/refs/tags/2023-03-01.tar.gz
    URL_HASH SHA256=7a9a4824958586980926a300b4717202485c4b4115ac031822e29aa4ef207e48
)
FetchContent_MakeAvailable(re2)

# Constrained decoding of tool calls: xgrammar, the grammar engine vLLM, SGLang and TensorRT-LLM use.
# Only its C++ core is needed. Its own CMake project unconditionally configures TVM-FFI Python
# bindings, so the sources are fetched populate-only (SOURCE_SUBDIR has no CMakeLists.txt) and the
# library is defined here. The release tarball leaves out the dlpack submodule; it is fetched at the
# commit v0.2.6 pins.
FetchContent_Declare(
    xgrammar
    URL      https://github.com/mlc-ai/xgrammar/archive/refs/tags/v0.2.6.tar.gz
    URL_HASH SHA256=f56889426b843aa0b52e3e8202a6c690a9d4cbd86df6afdc8be5d2428a7a57c2
    SOURCE_SUBDIR populate-only
)
FetchContent_Declare(
    dlpack
    URL      https://github.com/dmlc/dlpack/archive/bbd2f4d32427e548797929af08cfe2a9cbb3cf12.tar.gz
    URL_HASH SHA256=f5dcb30f8a3d1a41d48b5d5b3fe1631dfce11de76f9e7e5bd22c6dc75a758885
    SOURCE_SUBDIR populate-only
)
FetchContent_MakeAvailable(xgrammar dlpack)
file(GLOB_RECURSE XGRAMMAR_SOURCES CONFIGURE_DEPENDS ${xgrammar_SOURCE_DIR}/cpp/*.cc)
list(FILTER XGRAMMAR_SOURCES EXCLUDE REGEX ".*/cpp/tvm_ffi/.*")
add_library(xgrammar STATIC ${XGRAMMAR_SOURCES})
target_include_directories(xgrammar PUBLIC ${xgrammar_SOURCE_DIR}/include)
target_include_directories(xgrammar SYSTEM PUBLIC ${xgrammar_SOURCE_DIR}/3rdparty/picojson ${dlpack_SOURCE_DIR}/include)
target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
target_compile_features(xgrammar PUBLIC cxx_std_17)
set_target_properties(xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(sparkinfer_server
    src/image_input.cpp
    src/video_input.cpp
    src/sparkinfer_server.cpp
    src/model_engine.cpp
    src/chat_tokenizer.cpp
    src/chat_tools.cpp
    src/tool_grammar.cpp
    src/lmstudio_api.cpp
    src/ollama_api.cpp
    src/anthropic_api.cpp
    src/responses_api.cpp
)

target_include_directories(sparkinfer_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/examples
    # kernels/deterministic.h -- header-only, for the SPARKINFER_DETERMINISTIC banner line.
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/include
)

target_link_libraries(sparkinfer_server PRIVATE
    xgrammar
    sparkinfer_runtime
    tokenizers_cpp
    nlohmann_json::nlohmann_json
    re2::re2
    CUDA::cudart
    Threads::Threads
)

target_compile_features(sparkinfer_server PRIVATE cxx_std_17)

if(BUILD_TESTS)
    add_executable(chat_tools_test
        tests/chat_tools_test.cpp
        src/chat_tools.cpp
    )
    target_include_directories(chat_tools_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(chat_tools_test PRIVATE
        nlohmann_json::nlohmann_json
        re2::re2
    )
    target_compile_definitions(chat_tools_test PRIVATE
        SPARKINFER_SERVER_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    target_compile_features(chat_tools_test PRIVATE cxx_std_17)
    add_test(NAME chat_tools_test COMMAND chat_tools_test)

    # Constrained tool calls: every output the grammar admits must parse. CPU-only, no model.
    add_executable(tool_grammar_test
        tests/tool_grammar_test.cpp
        src/chat_tools.cpp
        src/tool_grammar.cpp
    )
    target_include_directories(tool_grammar_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    )
    target_link_libraries(tool_grammar_test PRIVATE
        nlohmann_json::nlohmann_json
        re2::re2
        xgrammar
    )
    target_compile_features(tool_grammar_test PRIVATE cxx_std_17)
    add_test(NAME tool_grammar_test COMMAND tool_grammar_test)

    # LM Studio /api/v0 response shaping: pure JSON over plain structs, no engine/CUDA, so it
    # runs on any box. The assertions are about exact field names and value vocabularies, which
    # is the entire contract with an LM Studio client.
    add_executable(lmstudio_api_test
        tests/lmstudio_api_test.cpp
        src/lmstudio_api.cpp
    )
    target_include_directories(lmstudio_api_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(lmstudio_api_test PRIVATE nlohmann_json::nlohmann_json)
    target_compile_features(lmstudio_api_test PRIVATE cxx_std_17)
    add_test(NAME lmstudio_api_test COMMAND lmstudio_api_test)

    # Ollama /api/* translation: exact field names, NANOSECOND durations, and the shape
    # differences from OpenAI (message.content, done, done_reason) that are easy to get wrong
    # while still emitting valid JSON.
    add_executable(ollama_api_test
        tests/ollama_api_test.cpp
        src/ollama_api.cpp
    )
    target_include_directories(ollama_api_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(ollama_api_test PRIVATE nlohmann_json::nlohmann_json)
    target_compile_features(ollama_api_test PRIVATE cxx_std_17)
    add_test(NAME ollama_api_test COMMAND ollama_api_test)

    # Anthropic /v1/messages translation: block types and indices, stream event order, stop_reason
    # vocabulary, and tool results moving from inside a user turn to OpenAI's tool messages.
    add_executable(anthropic_api_test
        tests/anthropic_api_test.cpp
        src/anthropic_api.cpp
    )
    target_include_directories(anthropic_api_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(anthropic_api_test PRIVATE nlohmann_json::nlohmann_json)
    target_compile_features(anthropic_api_test PRIVATE cxx_std_17)
    add_test(NAME anthropic_api_test COMMAND anthropic_api_test)

    # OpenAI /v1/responses translation: item types and ids, strictly increasing sequence numbers,
    # and the stateless refusals (previous_response_id, conversation, item_reference).
    add_executable(responses_api_test
        tests/responses_api_test.cpp
        src/responses_api.cpp
    )
    target_include_directories(responses_api_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(responses_api_test PRIVATE nlohmann_json::nlohmann_json)
    target_compile_features(responses_api_test PRIVATE cxx_std_17)
    add_test(NAME responses_api_test COMMAND responses_api_test)

    # Image input: base64, data: URL parsing, decode -- and the refusals, which are the
    # security-relevant half (no remote fetch, bounded payloads, no silent truncation).
    add_executable(image_input_test
        tests/image_input_test.cpp
        src/image_input.cpp
    )
    target_include_directories(image_input_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_compile_features(image_input_test PRIVATE cxx_std_17)
    add_test(NAME image_input_test COMMAND image_input_test)

    # Video input: the ffmpeg subprocess path against real containers ffmpeg synthesizes itself,
    # plus the same refusals image_input_test covers. SKIPS (passes) when ffmpeg is absent --
    # video decoding is an optional runtime dependency and must not turn a box's suite red for
    # not having it.
    add_executable(video_input_test
        tests/video_input_test.cpp
        src/video_input.cpp
        src/image_input.cpp
    )
    target_include_directories(video_input_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_compile_features(video_input_test PRIVATE cxx_std_17)
    add_test(NAME video_input_test COMMAND video_input_test)

    add_executable(chat_tokenizer_test
        tests/chat_tokenizer_test.cpp
        src/chat_tokenizer.cpp
        src/chat_tools.cpp
    )
    target_include_directories(chat_tokenizer_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(chat_tokenizer_test PRIVATE
        tokenizers_cpp
        nlohmann_json::nlohmann_json
        re2::re2
    )
    target_compile_features(chat_tokenizer_test PRIVATE cxx_std_17)
    add_test(NAME chat_tokenizer_test COMMAND chat_tokenizer_test)
endif()

install(TARGETS sparkinfer_server RUNTIME DESTINATION bin)
