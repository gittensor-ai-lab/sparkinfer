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

add_executable(sparkinfer_server
    src/image_input.cpp
    src/sparkinfer_server.cpp
    src/model_engine.cpp
    src/chat_tokenizer.cpp
    src/chat_tools.cpp
)

target_include_directories(sparkinfer_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/examples
    # kernels/deterministic.h -- header-only, for the SPARKINFER_DETERMINISTIC banner line.
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/include
)

target_link_libraries(sparkinfer_server PRIVATE
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
