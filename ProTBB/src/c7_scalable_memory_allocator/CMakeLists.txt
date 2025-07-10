set(SOURCES
    1_allocator_benchmark
    2_allocator_benchmark
    3_malloc_profile
    4_malloc_profile
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb TBB::tbbmalloc)
    # Below is for enabling TBB allocator in the second benchmark
    if(src STREQUAL "2_allocator_benchmark")
        target_compile_definitions(${src} PRIVATE USE_TBB_ALLOCATOR)
    elseif(src STREQUAL "4_malloc_profile")
        target_link_libraries(${src} TBB::tbbmalloc_proxy)
    endif()
endforeach()