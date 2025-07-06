set(SOURCES
    1_flow_grapg_basic_two_node
    2_join_node_example
    3_try_put_vs_input_node
    4_flow_graph_serial_unlimited_tagmatching_demo
    5_dependency_graph_continue_msg
)

foreach(src ${SOURCES})
    add_executable(${src} ${src}.cc)
    target_link_libraries(${src} TBB::tbb)
endforeach()