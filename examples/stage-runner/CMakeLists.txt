set(TARGET llama-stage-runner)
add_executable(${TARGET} stage-runner.cpp)
install(TARGETS ${TARGET} RUNTIME)
# Driver uses only public llama.h / ggml.h APIs (no common_* helpers), so it does
# not link `common`. `llama` brings in `ggml` transitively (ggml_backend_*, etc.).
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
