file(GLOB SOURCES "*.cc")

foreach(src ${SOURCES})
    get_filename_component(exec_name ${src} NAME_WE)
    add_executable(${exec_name} ${src})
    target_link_libraries(${exec_name} TBB::tbb)
endforeach()