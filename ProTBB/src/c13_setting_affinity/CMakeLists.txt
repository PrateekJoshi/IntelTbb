set(SOURCES
    1_pinning_threads_to_cores
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()