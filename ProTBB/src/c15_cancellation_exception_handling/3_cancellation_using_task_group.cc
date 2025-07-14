// Pg 393
#include <tbb/task_group.h>
#include <tbb/tick_count.h>

#include <iostream>
#include <vector>

int grainsize = 100;
std::vector<int> data;
int myindex = -1;
tbb::task_group g;

void SerialSearch(long begin, long end)
{
  for (int i = begin; i < end; ++i)
  {
    if (data[i] == -2)
    {
      myindex = i;
      g.cancel();
      break;
    }
  }
}

void ParallelSearch(long begin, long end)
{
  // tbb::is_current_task_group_canceling()) calls the equivalent of
  // <group we are in>.context().is_group_execution_cancelled()
  // which is not accessible directly since the context method is
  // protected (private to the class)
  if (tbb::is_current_task_group_canceling()) return;
  if ((end - begin) < grainsize)
  {  // cutoof equivalent
    return SerialSearch(begin, end);
  }
  else
  {
    long mid = begin + (end - begin) / 2;
    g.run([&] { ParallelSearch(begin, mid); });  // spawn a task
    g.run([&] { ParallelSearch(mid, end); });    // spawn another task
  }
}

int main(int argc, char** argv)
{
  int n = 100000000;
  data.resize(n);
  data[n / 2] = -2;

  auto t0 = tbb::tick_count::now();
  SerialSearch(0, n);
  auto t1 = tbb::tick_count::now();
  ParallelSearch(0, n);
  g.wait();  // wait for all spawned tasks
  auto t2 = tbb::tick_count::now();
  double t_s = (t1 - t0).seconds();
  double t_p = (t2 - t1).seconds();

  std::cout << "SerialSearch:   " << myindex << " Time: " << t_s << std::endl;
  std::cout << "ParallelSearch: " << myindex << " Time: " << t_p << " Speedup: " << t_s / t_p << std::endl;
  return 0;
}