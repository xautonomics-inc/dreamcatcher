set(TARGET llama-stage-runner)
add_executable(${TARGET} stage-runner.cpp)
install(TARGETS ${TARGET} RUNTIME)
# Driver uses only public llama.h / ggml.h APIs (no common_* helpers), so it does
# not link `common`. `llama` brings in `ggml` transitively (ggml_backend_*, etc.).
# Server role (--role server, stage-server.h): vendored SPLIT cpp-httplib (declaration-only
# header + static target built from vendor/cpp-httplib — must LINK it) + nlohmann json via
# the vendor include root.
target_link_libraries(${TARGET} PRIVATE llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_compile_definitions(${TARGET} PRIVATE STAGE_SERVER)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Expert-tensor disaggregation (P1): the expert-server role and the
# byte-exactness / overhead harness. Both are additive - the llama-stage-runner
# target above is untouched.

set(TARGET llama-expert-server)
add_executable(${TARGET} expert-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
# Talks ggml directly (no llama model load) but needs the wire constants from
# src/llama-experts-remote.h.
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-expert-check)
add_executable(${TARGET} expert-check.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
