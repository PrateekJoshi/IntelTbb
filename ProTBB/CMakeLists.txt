cmake_minimum_required(VERSION 3.14...3.22)

# ---- Project ----

# Note: update this to your new project's name and version
project(
  ProTBB
  VERSION 1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(cmake/CPM.cmake)
include(cmake/tools.cmake)
include(cmake/ClangFormat.cmake)

# Code coverage enabled ?
option(CODE_COVERAGE "Enable coverage reporting" ON)

# add "-V" in ctest command to see full logs of gtest
if(CODE_COVERAGE)
    include(CTest)
    include(cmake/CodeCoverage.cmake)
    APPEND_COVERAGE_COMPILER_FLAGS()
    SETUP_TARGET_FOR_COVERAGE_LCOV(
        NAME coverage                 
        EXECUTABLE ctest -j ${n_cores} ./src # Executable in PROJECT_BINARY_DIR
    )
endif()

# needed to generate test target
enable_testing()

# ---- Find dependencies ----
find_package(TBB REQUIRED)

# ---- Add subdirectories ----
add_subdirectory(src)
add_subdirectory(documentation)