# Minimum CMake version
cmake_minimum_required(VERSION 3.30)

# Project name and language
project(Xalm LANGUAGES C CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build directories
set(SRC_DIR "src")
set(VENDOR_DIR "vendor")

# Compiler flags
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS
    link_directories(/opt/homebrew/opt/libomp/lib)
    include_directories(/opt/homebrew/opt/libomp/include)
else()
    # Linux
    link_directories(/usr/lib/x86_64-linux-gnu)
    include_directories(/usr/include)
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add compile options
add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wpointer-arith
        -march=native
        -fcolor-diagnostics
)

# Additional optimizations for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# Find source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.cc"
        "${SRC_DIR}/*.cpp"
        "${VENDOR_DIR}/*.c"
        "${VENDOR_DIR}/*.cc"
        "${VENDOR_DIR}/*.cpp"
)

# Define main executable
add_executable(main ${SOURCES})
target_include_directories(main PRIVATE ${OpenMP_INCLUDE_DIRS})
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/vendor)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Clean target
add_custom_target(clean-all
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning build directory"
)