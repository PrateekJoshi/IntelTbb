// Pg 418
#include <tbb/tbb.h>

#include <iostream>
#include <string>

static inline void spinWaitForAtLeast(double sec = 0.0);

static inline double executeFor(int num_trials, int N, double tpi)
{
  tbb::tick_count t0;
  for (int t = -1; t < num_trials; ++t)
  {
    if (!t) t0 = tbb::tick_count::now();
    for (int i = 0; i < N; ++i)
    {
      spinWaitForAtLeast(tpi);
    }
  }
  tbb::tick_count t1 = tbb::tick_count::now();
  return (t1 - t0).seconds() / num_trials;
}

template <typename P>
static inline double executePfor(int num_trials, int N, int gs, P& p, double tpi)
{
  tbb::tick_count t0;
  for (int t = -1; t < num_trials; ++t)
  {
    if (!t) t0 = tbb::tick_count::now();
    tbb::parallel_for(
        tbb::blocked_range<int>{0, N, static_cast<size_t>(gs)},
        [tpi](const tbb::blocked_range<int>& r)
        {
          for (int i = r.begin(); i < r.end(); ++i)
          {
            spinWaitForAtLeast(tpi);
          }
        },
        p);
  }
  tbb::tick_count t1 = tbb::tick_count::now();
  return (t1 - t0).seconds() / num_trials;
}

int main()
{
  // use the most performance codes
  // only a single NUMA node
  // and only 1 thread per core
  tbb::task_arena::constraints c;
  c.set_numa_id(tbb::info::numa_nodes()[0]);
  c.set_core_type(tbb::info::core_types().back());
  c.set_max_threads_per_core(1);
  c.set_max_concurrency(std::min(8, tbb::info::default_concurrency(c)));
  tbb::task_arena a(c);

  std::cout << "Using an arena with " << a.max_concurrency() << " slots\n";

  a.execute(
      [&]()
      {
        tbb::auto_partitioner auto_p;
        tbb::simple_partitioner simple_p;
        tbb::static_partitioner static_p;
        const std::string pname[4] = {"simple", "auto", "affinity", "static"};

        //    const int N = 262144;
        const int N = 1024;
        const int T = 20;
        const double ten_ns = 0.00000001;
        const double twenty_us = 0.00002;
        double timing[4][19];

        for (double tpi = ten_ns; tpi < twenty_us; tpi *= 10)
        {
          std::cout << "Speedups for " << tpi << " seconds per iteration" << std::endl << "partitioner";
          for (int gs = 1, i = 0; gs <= N; gs *= 2, ++i) std::cout << ", " << gs;
          std::cout << std::endl;

          double serial_time = executeFor(T, N, tpi);

          for (int gs = 1, i = 0; gs <= N; gs *= 2, ++i)
          {
            tbb::affinity_partitioner affinity_p;
            spinWaitForAtLeast(0.001);
            timing[0][i] = executePfor(T, N, gs, simple_p, tpi);
            timing[1][i] = executePfor(T, N, gs, auto_p, tpi);
            timing[2][i] = executePfor(T, N, gs, affinity_p, tpi);
            timing[3][i] = executePfor(T, N, gs, static_p, tpi);
          }
          for (int p = 0; p < 4; ++p)
          {
            std::cout << pname[p];
            for (int gs = 1, i = 0; gs <= N; gs *= 2, ++i) std::cout << ", " << serial_time / timing[p][i];
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }
      });

  return 0;
}

static inline void spinWaitForAtLeast(double sec)
{
  if (sec == 0.0) return;
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}