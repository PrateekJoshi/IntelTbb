set(SOURCES
    1_coarse_grained_locking
    2_fine_grained_locking
    3_fine_grained_using_atomics
    4_hist_tls_reduction
    5_hist_tls_reduction_moderncpp
    6_hist_tls_reduction_combine_each
    7_hist_tls_combinable_reduction_combine_each
    8_hist_tls_reduction
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()