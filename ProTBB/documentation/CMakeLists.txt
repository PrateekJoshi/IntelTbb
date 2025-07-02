cmake_minimum_required(VERSION 3.14...3.22)

project(UCXDocs)

# ---- Doxygen variables ----

# Find Doxygen executable
find_package(Doxygen REQUIRED)

# set Doxyfile variables
set(DOXYGEN_PROJECT_NAME "MyAppName")
set(DOXYGEN_PROJECT_VERSION "1.1")
set(DOXYGEN_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

# Configure the Doxyfile
# Make sure your Doxyfile template (Doxyfile in the current directory) is set up
# for vanilla Doxygen output.
configure_file(${CMAKE_CURRENT_LIST_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(
  GenerateDocs
  ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
  COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
  COMMAND ${CMAKE_COMMAND} -E echo "Docs written to: ${DOXYGEN_OUTPUT_DIRECTORY}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)