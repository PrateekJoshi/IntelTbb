# Apply linker flags globally
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--verbose")
# set(CMAKE_BUILD_TYPE Release)
add_subdirectory(c1_jumping_right_in)
add_subdirectory(c2_generic_parallel_algo)
add_subdirectory(c3_flow_graphs)
# add_subdirectory(c4_tbb_parallel_algo_std_cpp)