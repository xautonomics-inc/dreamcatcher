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
