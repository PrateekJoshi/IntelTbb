// Pg 389
/*
With Cancellation:

(conanenv) prateek@prateek:~/Workspace/IntelTbb/build/src/c15_cancellation_exception_handling$ ./1_cancellation 
Index 500 found in 0.00776095 seconds!

Without Cancellation:
(conanenv) prateek@prateek:~/Workspace/IntelTbb/build/src/c15_cancellation_exception_handling$ ./1_cancellation 
Index 500 found in 0.27683 seconds!

Speedup (S) = Execution Time (Before) / Execution Time (After) = 0.27683/0.00776095 = 35.7 faster 🏎️

*/
#include <tbb/tbb.h>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
  int n = 1000000000;
  size_t nth = 4;
  tbb::global_control global_limit{tbb::global_control::max_allowed_parallelism, nth};

  std::vector<int> data(n);
  data[500] = -2;
  int index = -1;
  auto t1 = tbb::tick_count::now();
  tbb::parallel_for(tbb::blocked_range<int>{0, n},
    [&](const tbb::blocked_range<int>& r){
        for(int i=r.begin(); i!=r.end(); ++i){
          if(data[i] == -2) {
            index = i;
            // commenting out the following line, can increase run time
            tbb::task::current_context()->cancel_group_execution();
            break;
          }
        }
  });
  auto t2 = tbb::tick_count::now();
  std::cout << "Index "     << index;
  std::cout << " found in " << (t2-t1).seconds() << " seconds!\n";
  return 0;
}