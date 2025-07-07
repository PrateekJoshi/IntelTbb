set(SOURCES
    1_tbb_custom_iterators
    2_parallel_stl
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()