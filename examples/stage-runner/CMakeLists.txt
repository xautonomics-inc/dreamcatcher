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

# RoCEv2 (Scope-B) forward-path option for the ring roles. stage_rdma.cpp is a
# self-contained TU: it includes ggml-rpc/transport.h (socket_t) but NOT httplib,
# so the runner's httplib `socket_t = int` never collides (the engine boundary is
# the whole-file translation-unit split). It reaches the socket_t symbols through
# the ggml-transport OBJECT library — NOT by compiling transport.cpp a second
# time, which under the per-source GGML_RPC_RDMA define would produce a TCP-only
# ABI mismatch with the RDMA-enabled transport inside libggml. Default OFF: with
# the option off the runner is byte-identical to before (RDMA-off parity gate).
include(CMakeDependentOption)
cmake_dependent_option(STAGE_RDMA_TRANSPORT
    "stage-runner: RoCEv2 (Scope-B) forward path via ggml-rpc transport"
    OFF "GGML_RPC;TARGET ggml-transport" OFF)
if (STAGE_RDMA_TRANSPORT)
    target_compile_definitions(${TARGET} PRIVATE STAGE_RDMA_TRANSPORT=1)
    target_sources(${TARGET} PRIVATE stage_rdma.cpp)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(${TARGET} PRIVATE ggml-transport)
endif()

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
