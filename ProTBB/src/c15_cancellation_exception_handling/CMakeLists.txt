set(SOURCES
    1_cancellation
    2_cancellation_alternate
    3_cancellation_using_task_group
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()