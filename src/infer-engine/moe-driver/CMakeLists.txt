# moe-driver — index-driven MoE expert-offload driver.
#
# Standalone static lib, added via add_subdirectory() from the llama.cpp build
# behind the JADE_MOE_DRIVER option (the seam delegates to it only when active).
# Two compile tiers so the host logic stays portable:
#   * always:        moe_policy.cpp + moe_driver.cpp  (ggml/CUDA-free host core)
#   * when ggml present: + graph.cpp                  (the build_moe_ffn paging hook)
#
# Standalone (host-only) build for unit tests on any platform:
#   cmake -S . -B build -DMOE_DRIVER_STANDALONE=ON && cmake --build build && ./build/moe_driver_test

cmake_minimum_required(VERSION 3.18)
project(moe_driver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MOE_DRIVER_STANDALONE "Build host-only + tests, without ggml/CUDA graph hook" OFF)

set(MOE_DRIVER_SOURCES
    src/moe_policy.cpp
    src/moe_driver.cpp)

# The graph hook needs ggml headers; only compile it inside the llama.cpp build
# (where the `ggml` target exists). Host-only/standalone builds omit it.
if(NOT MOE_DRIVER_STANDALONE AND TARGET ggml)
    list(APPEND MOE_DRIVER_SOURCES src/graph.cpp)
endif()

add_library(moe_driver STATIC ${MOE_DRIVER_SOURCES})
target_include_directories(moe_driver PUBLIC include)
target_compile_options(moe_driver PRIVATE -Wall -Wextra)

if(NOT MOE_DRIVER_STANDALONE AND TARGET ggml)
    # graph.cpp emits ggml ops + issues host->VRAM paging copies.
    target_link_libraries(moe_driver PUBLIC ggml)
    target_compile_definitions(moe_driver PRIVATE JADE_MOE_GRAPH=1)
endif()

# Host-side unit test (policy + C-ABI lifecycle). No ggml/CUDA needed.
if(MOE_DRIVER_STANDALONE)
    add_executable(moe_driver_test test/moe_policy_test.cpp)
    target_link_libraries(moe_driver_test PRIVATE moe_driver)
    enable_testing()
    add_test(NAME moe_driver_test COMMAND moe_driver_test)
endif()
