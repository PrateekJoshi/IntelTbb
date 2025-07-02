add_executable(1_hello_tbb 1_hello_tbb.cc )
target_link_libraries(1_hello_tbb TBB::tbb)

add_executable(2_hello_parallel_stl 2_hello_parallel_stl.cc )
target_link_libraries(2_hello_parallel_stl TBB::tbb)

add_executable(3_gamma_baseline 3_gamma_baseline.cc )
target_link_libraries(3_gamma_baseline TBB::tbb)


add_executable(4_gamma_opt_tbb_flowgraph_pipelined 4_gamma_opt_tbb_flowgraph_pipelined.cc )
target_link_libraries(4_gamma_opt_tbb_flowgraph_pipelined TBB::tbb)


add_executable(5_gamma_opt_parallel_for 5_gamma_opt_parallel_for.cc )
target_link_libraries(5_gamma_opt_parallel_for TBB::tbb)

add_executable(6_gamma_opt_parallel_for_transform 6_gamma_opt_parallel_for_transform.cc )
target_link_libraries(6_gamma_opt_parallel_for_transform TBB::tbb)