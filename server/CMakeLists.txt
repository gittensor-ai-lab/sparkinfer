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

add_executable(sparkinfer_server
    src/sparkinfer_server.cpp
    src/model_engine.cpp
    src/chat_tokenizer.cpp
)

target_include_directories(sparkinfer_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/examples
)

target_link_libraries(sparkinfer_server PRIVATE
    sparkinfer_runtime
    tokenizers_cpp
    CUDA::cudart
    Threads::Threads
)

target_compile_features(sparkinfer_server PRIVATE cxx_std_17)

install(TARGETS sparkinfer_server RUNTIME DESTINATION bin)
