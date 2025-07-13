set(SOURCES
    1_global_control_and_explicit_arena
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb TBB::tbbmalloc)
endforeach()