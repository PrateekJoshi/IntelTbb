set(SOURCES
    1_lambda_vs_functors
    2_parallel_invoke_two_quickshort
    3_parallel_for_matrix_multipy
    4_parallel_reduce_maxNum
    5_parallel_reduce_pi
    6_parallel_scan_running_sum
    7_parallel_foreach_prime
    8_parallel_pipeline
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()