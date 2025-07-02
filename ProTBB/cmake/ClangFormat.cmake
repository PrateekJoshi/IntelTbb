# Find clang-format executable
find_program(CLANG_FORMAT_EXE clang-format)
if(NOT CLANG_FORMAT_EXE)
    message(FATAL_ERROR "clang-format not found. Please install it (e.g., apt-get install clang-format) and ensure it's in your PATH.")
endif()

# Collect all C++ source files
set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
file(GLOB_RECURSE CXX_SOURCE_FILES
    "${PROJECT_ROOT_DIR}/src/*.cpp"
    "${PROJECT_ROOT_DIR}/src/*.cxx"
    "${PROJECT_ROOT_DIR}/src/*.cc"
    "${PROJECT_ROOT_DIR}/src/*.h"
    "${PROJECT_ROOT_DIR}/src/*.hpp"
    "${PROJECT_ROOT_DIR}/src/*.hxx"
)

message(STATUS "clangd-format WORKING_DIRECTORY : " ${PROJECT_ROOT_DIR})

# Create a custom target to format all C++ files
add_custom_target(format_cpp
    COMMAND ${CLANG_FORMAT_EXE} -i ${CXX_SOURCE_FILES}
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR}
    COMMENT "Formatting all C++ files..."
)

# Optional: Add a check target for CI/CD
add_custom_target(check_cpp_format
    COMMAND ${CLANG_FORMAT_EXE} --Werror --dry-run ${CXX_SOURCE_FILES}
    WORKING_DIRECTORY ${PROJECT_ROOT_DIR}
    COMMENT "Checking C++ file formatting..."
    # Ensure this target fails if formatting is not applied
    set_property(TARGET check_cpp_format PROPERTY JOB_POOL EXCLUSIVE)
)