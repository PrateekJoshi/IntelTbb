set(SOURCES
    1_parallel_invoke_fib
    2_task_group_fib
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb TBB::tbbmalloc)
endforeach()