cmake_minimum_required(VERSION 3.20)
project(FlashAttentionMetal VERSION 1.0 LANGUAGES CXX OBJCXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Objective-C++ Standard
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Only build on macOS
if(NOT APPLE)
    message(FATAL_ERROR "Flash Attention Metal backend only supports macOS with Apple Silicon")
endif()

# Check for Apple Silicon
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCHITECTURE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT ARCHITECTURE STREQUAL "arm64")
    message(WARNING "Metal backend is optimized for Apple Silicon (arm64), but detected: ${ARCHITECTURE}")
endif()

message(STATUS "Building Flash Attention Metal for ${ARCHITECTURE}")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp)

# Find required Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "Found Foundation framework: ${FOUNDATION_FRAMEWORK}")

# Find Python and PyTorch for bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python3_EXECUTABLE}")
message(STATUS "Python version: ${Python3_VERSION}")

# Find pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PATH})
endif()

# Try to find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_CMAKE_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
    find_package(Torch QUIET)
endif()

# Metal Shader Compilation
set(METAL_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attention_fwd.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attention_bwd.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.metal
)

set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/flash_attn_kernels.metallib)

# Check if metal compiler is available
find_program(METAL_COMPILER metal)

if(METAL_COMPILER)
    # Compile Metal shaders to metallib
    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADERS} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${METALLIB_OUTPUT}
        DEPENDS ${METAL_SHADERS}
        COMMENT "Compiling Metal shaders to metallib"
        VERBATIM
    )

    add_custom_target(metal_shaders ALL DEPENDS ${METALLIB_OUTPUT})
    message(STATUS "Metal compiler found - will compile shaders")
else()
    message(WARNING "Metal compiler not found - skipping shader compilation")
    message(WARNING "Install Xcode: xcode-select --install")
    add_custom_target(metal_shaders)
endif()

# C++ Library - Core Flash Attention Metal Implementation
set(FLASH_ATTN_METAL_SOURCES
    src/flash_attn_core.cpp
)

add_library(flash_attn_metal_core SHARED ${FLASH_ATTN_METAL_SOURCES})

target_include_directories(flash_attn_metal_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp
)

target_link_libraries(flash_attn_metal_core PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
)

# Ensure shaders are compiled before library
add_dependencies(flash_attn_metal_core metal_shaders)

# Install metallib alongside the library
install(FILES ${METALLIB_OUTPUT} DESTINATION lib)

# Optional: Link PyTorch if found
if(Torch_FOUND)
    target_link_libraries(flash_attn_metal_core PUBLIC torch)
    target_compile_definitions(flash_attn_metal_core PRIVATE WITH_TORCH)
    message(STATUS "Linking with PyTorch")
endif()

# Python Bindings with pybind11
# First try to find via standard CMake
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # Try using the Python module location
    message(STATUS "Trying to find pybind11 via Python...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )
    
    if(PYBIND11_RESULT EQUAL 0 AND PYBIND11_DIR)
        message(STATUS "Found pybind11 CMake dir: ${PYBIND11_DIR}")
        set(pybind11_DIR ${PYBIND11_DIR})
        find_package(pybind11 REQUIRED CONFIG)
    else()
        message(STATUS "Could not find pybind11 via Python")
    endif()
endif()

if(pybind11_FOUND)
    message(STATUS "✓ Found pybind11, building Python bindings")
    
    pybind11_add_module(_flash_attn_metal python/bindings.cpp)
    
    target_link_libraries(_flash_attn_metal PRIVATE flash_attn_metal_core)
    
    target_include_directories(_flash_attn_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp
    )
    
    # Set output directory for Python module
    set_target_properties(_flash_attn_metal PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
else()
    message(WARNING "pybind11 not found, Python bindings will not be built. Install with: pip install pybind11")
endif()

# Installation
install(TARGETS flash_attn_metal_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/flash_attn_metal
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Testing (optional)
option(BUILD_TESTS "Build C++ tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        add_executable(test_metal_device tests/test_metal_device.cpp)
        target_link_libraries(test_metal_device flash_attn_metal_core GTest::GTest GTest::Main)
        add_test(NAME test_metal_device COMMAND test_metal_device)
    else()
        message(WARNING "GTest not found, C++ tests will not be built")
    endif()
endif()

# Print summary
message(STATUS "")
message(STATUS "=================== Configuration Summary ===================")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "Architecture:         ${ARCHITECTURE}")
message(STATUS "C++ compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message(STATUS "Python:               ${Python3_EXECUTABLE}")
message(STATUS "PyTorch found:        ${Torch_FOUND}")
message(STATUS "pybind11 found:       ${pybind11_FOUND}")
message(STATUS "Metal shaders:        ${METAL_SHADERS}")
message(STATUS "Output metallib:      ${METALLIB_OUTPUT}")
message(STATUS "=============================================================")
message(STATUS "")
