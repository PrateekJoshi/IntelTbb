set(SOURCES
    1_task_arena_isolate
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()