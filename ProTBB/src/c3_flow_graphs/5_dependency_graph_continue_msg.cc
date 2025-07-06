// Pg 105
/*
🧠 How It Works
continue_node is used to represent tasks that don't require input data, just a signal to start.

continue_msg is a lightweight message used to trigger execution.

The graph structure defines task dependencies, not data flow.

finalTask will only run after both taskA and taskB complete.

        start
        /   \
     taskA  taskB
        \   /
      finalTask
*/
#include <tbb/flow_graph.h>
#include <iostream>

using namespace tbb::flow;

int main() {
    // Step 1: Create a flow graph
    graph g;

    // Step 2: Create continue_nodes that perform simple tasks
    continue_node<continue_msg> start(g, [](const continue_msg&) {
        std::cout << "Start task\n";
    });

    continue_node<continue_msg> taskA(g, [](const continue_msg&) {
        std::cout << "Task A completed\n";
    });

    continue_node<continue_msg> taskB(g, [](const continue_msg&) {
        std::cout << "Task B completed\n";
    });

    continue_node<continue_msg> finalTask(g, [](const continue_msg&) {
        std::cout << "Final task after A and B\n";
    });

    // Step 3: Define dependencies using make_edge
    // start → taskA
    // start → taskB
    // taskA → finalTask
    // taskB → finalTask
    make_edge(start, taskA);
    make_edge(start, taskB);
    make_edge(taskA, finalTask);
    make_edge(taskB, finalTask);

    // Step 4: Trigger the graph by sending a continue_msg to the start node
    start.try_put(continue_msg{});

    // Step 5: Wait for all tasks to complete
    g.wait_for_all();

    return 0;
}
