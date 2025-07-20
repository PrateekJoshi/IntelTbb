// Pg 588
// Benchmarking command :
// likwid-bench -t stream -i 1 -w S0:12GB:16-0:S0,1:S0,2:S0
/*
 - S0: The threads are pinned to the NUMA node 0.
 - 12 GB: The three triad arrays occupy 12 GB (4 GB per array).
 - 16: 16 threads will share the computation, each one processing
 chunks of 31,250,000 doubles (this is, 4000 million bytes/8 bytes per
 double/16 threads).
 - 0:S0,1:S0,2:S0: The three arrays are allocated on the NUMA node 0.
*/
#include <tbb/tbb.h>

#include <cstdio>
#include <iostream>
// #include <asm/cachectl.h>

int main(int argc, const char* argv[])
{
  int nth = 4;
  size_t vsize = 100000000;
  float alpha = 0.5;
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nth);

  std::unique_ptr<double[]> A{new double[vsize]};
  std::unique_ptr<double[]> B{new double[vsize]};
  std::unique_ptr<double[]> C{new double[vsize]};

  for (size_t i = 0; i < vsize; i++)
  {
    A[i] = B[i] = i;
  }
  // cacheflush((char*)A, vsize*sizeof(double), DCACHE);

  auto t = tbb::tick_count::now();
  tbb::parallel_for(tbb::blocked_range<size_t>{0, vsize},
                    [&](const tbb::blocked_range<size_t>& r)
                    {
                      for (size_t i = r.begin(); i < r.end(); ++i) C[i] = A[i] + alpha * B[i];
                    });
  double ts = (tbb::tick_count::now() - t).seconds();

#ifdef VERBOSE
  std::cout << "Results: " << '\n';
  for (size_t i = 0; i < vsize; i++)
  {
    std::cout << C[i] << ", ";
  }
  std::cout << '\n';
#endif

  std::cout << "Time: " << ts << " seconds; ";
  std::cout << "Bandwidth: " << vsize * 24 / ts / 1000000.0 << "MB/s\n";
  return 0;
}