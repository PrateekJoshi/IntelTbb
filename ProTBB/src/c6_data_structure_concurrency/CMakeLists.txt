set(SOURCES
    1_concurrent_hash_map
    2_concurrent_queues
    3_iterators
    4_concurrent_vector
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()