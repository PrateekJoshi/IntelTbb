/*
try_put:
========
✅ Pros
Simple and direct.
Easy to control when and how data is pushed.
❌ Cons
Not reactive or dynamic.
You must manage the loop and timing yourself.
Best for : Static input

input_node:
========
✅ Pros
Automatically generates and pushes data.
Great for streaming or dynamic sources.
Clean separation of data generation and processing.

❌ Cons
Slightly more complex to set up.
Less control over individual try_put calls.
Best for : Streaming input

source_node is deprecated in TBB 2020 and later, so use input_node instead.
*/

#include <tbb/tbb.h>
#include <iostream>

using namespace tbb::flow;

void try_put_demo()
{
    graph g;

    function_node<int> processor(g, unlimited, [](int x) {
        std::cout << "Processing: " << x << std::endl;
    });

    // Manually push data using a loop
    for (int i = 0; i < 5; ++i) {
        processor.try_put(i);
    }

    g.wait_for_all();
}

void input_node_demo()
{
    graph g;
    int count = 0;
    const int max_count = 5;

    input_node<int> input(g, [&](tbb::flow_control& fc) -> int {
        if (count < max_count) {
            return count++;
        } else {
            fc.stop();  // Signal no more data
            return 0;   // Return dummy value
        }
    });

    function_node<int> processor(g, unlimited, [](int x) {
        std::cout << "Processing: " << x << std::endl;
    });

    make_edge(input, processor);

    input.activate();  // Start the input node
    g.wait_for_all();
}

int main() {
    try_put_demo();
    input_node_demo();
    return 0;
}
