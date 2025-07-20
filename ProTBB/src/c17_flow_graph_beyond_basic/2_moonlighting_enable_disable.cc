// Pg 486
#include <tbb/tbb.h>
#include <iostream>

static inline void spinWaitForAtLeast(double sec)
{
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

void warmupTBB()
{
  tbb::parallel_for(0, tbb::this_task_arena::max_concurrency(), [](int) { spinWaitForAtLeast(0.001); });
}

void fig_17_23()
{
  int P = tbb::this_task_arena::max_concurrency();
  tbb::concurrent_vector<std::string> trace;
  double spin_time = 1e-3;
  tbb::flow::graph g;

  int src_cnt = 0;
  tbb::flow::input_node<int> source{g,
                                    [&src_cnt, P, spin_time](tbb::flow_control& fc) -> int
                                    {
                                      if (src_cnt < P)
                                      {
                                        int i = src_cnt++;
                                        spinWaitForAtLeast(spin_time);
                                        return i;
                                      }
                                      else
                                      {
                                        fc.stop();
                                        return 0;
                                      }
                                    }};

  tbb::flow::function_node<int> unlimited_node(
      g, tbb::flow::unlimited,
      [&trace, P, spin_time](int i)
      {
        int tid = tbb::this_task_arena::current_thread_index();
        trace.push_back(std::to_string(i) + " started by " + std::to_string(tid));
        tbb::parallel_for(0, P - 1, [spin_time](int i) { spinWaitForAtLeast((i + 1) * spin_time); });
        trace.push_back(std::to_string(i) + " completed by " + std::to_string(tid));
      });

  tbb::flow::make_edge(source, unlimited_node);
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
  s.set_name("source");
  n.set_name("unlimited_node");
#endif
  source.activate();
  g.wait_for_all();

  for (auto s : trace)
  {
    std::cout << s << std::endl;
  }
}

void noMoonlighting()
{
  int P = tbb::this_task_arena::max_concurrency();
  tbb::concurrent_vector<std::string> trace;
  double spin_time = 1e-3;
  tbb::flow::graph g;

  int src_cnt = 0;
  tbb::flow::input_node<int> source{g,
                                    [&src_cnt, P, spin_time](tbb::flow_control& fc) -> int
                                    {
                                      if (src_cnt < P)
                                      {
                                        int i = src_cnt++;
                                        spinWaitForAtLeast(spin_time);
                                        return i;
                                      }
                                      else
                                      {
                                        fc.stop();
                                        return 0;
                                      }
                                    }};

  tbb::flow::function_node<int> unlimited_node{
      g, tbb::flow::unlimited, [&trace, P, spin_time](int i)
      {
        int tid = tbb::this_task_arena::current_thread_index();
        trace.push_back(std::to_string(i) + " started by " + std::to_string(tid));
        tbb::this_task_arena::isolate(
            [P, spin_time]()
            { tbb::parallel_for(0, P - 1, [spin_time](int i) { spinWaitForAtLeast((i + 1) * spin_time); }); });
        trace.push_back(std::to_string(i) + " completed by " + std::to_string(tid));
      }};

  tbb::flow::make_edge(source, unlimited_node);
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
  s.set_name("s");
  n.set_name("n");
#endif
  source.activate();
  g.wait_for_all();

  for (auto s : trace)
  {
    std::cout << s << std::endl;
  }
}

int main()
{
  warmupTBB();
  std::cout << "Without isolation:" << std::endl;
  fig_17_23();
  spinWaitForAtLeast(10e-3);

  warmupTBB();
  std::cout << std::endl << "With isolation:" << std::endl;
  noMoonlighting();
  return 0;
}
