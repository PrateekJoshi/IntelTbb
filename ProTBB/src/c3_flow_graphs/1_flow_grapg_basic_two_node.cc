// Pg 83
#include <iostream>
#include <tbb/tbb.h>

void graphTwoNodes() {
  // step 1: construct the graph
  tbb::flow::graph g;
   
  // step 2: make the nodes
  tbb::flow::function_node<int, std::string> my_first_node{g, 
    tbb::flow::unlimited, 
    []( const int& in ) {
      std::cout << "first node received: " << in << std::endl; 
      return std::to_string(in);
    }
  };

  tbb::flow::function_node<std::string> my_second_node{g, 
    tbb::flow::unlimited, 
    []( const std::string& in ) {
      std::cout << "second node received: " << in << std::endl; 
    }
  };

  // step 3: add edges
  tbb::flow::make_edge(my_first_node, my_second_node);

  // step 4: send message that eagerly starts graph execution
  my_first_node.try_put(10);

  // step 5: wait for graph to complete
  g.wait_for_all();
}

static void warmupTBB();

int main(int argc, char *argv[]) {
  warmupTBB();
  double parallel_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    graphTwoNodes(); 
    parallel_time = (tbb::tick_count::now() - t0).seconds();
  }

  std::cout << "parallel_time == " << parallel_time << " seconds" << std::endl;
  return 0;
}

static void warmupTBB() {
  // This is a simple loop that should get workers started.
  // oneTBB creates workers lazily on first use of the library
  // so this hides the startup time when looking at trivial
  // examples that do little real work. 
  tbb::parallel_for(0, tbb::info::default_concurrency(), 
    [=](int) {
      tbb::tick_count t0 = tbb::tick_count::now();
      while ((tbb::tick_count::now() - t0).seconds() < 0.01);
    }
  );
}