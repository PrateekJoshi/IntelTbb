// Pg - 52
/*
To use a TBB parallel_reduce, we need to identify the range,
body, identity value, and reduction function. For this example, the range is [0, num_
intervals), and the body will be similar to the i-loop in Figure 2-11. The identity value
is 0.0 since we are performing a sum. And the reduction body, which needs to combine
partial results, will return the sum of two values. The parallel implementation using a
TBB parallel_reduce .
*/

#include <cmath>
#include <tbb/tbb.h>
//
// Estimating pi using numerical integration
//
double serialPI(int num_intervals) {
  double dx = 1.0 / num_intervals;
  double sum = 0.0;
  for (int i = 0; i < num_intervals; ++i) {
    double x = (i+0.5)*dx;
    double h = std::sqrt(1-x*x);
    sum += h*dx;
  }
  double pi = 4 * sum;
  return pi;
}

//
// Estimating pi using numerical integration
// with a TBB parallel_reduce
//
double parallelPI(int num_intervals) {
  double dx = 1.0 / num_intervals;
  double sum = tbb::parallel_reduce(
    /* range = */ tbb::blocked_range<int>(0, num_intervals), 
    /* identity = */ 0.0,
    /* func */ 
    [=](const tbb::blocked_range<int>& r, double init) -> double {
      for (int i = r.begin(); i != r.end(); ++i) {
        double x = (i + 0.5)*dx;
        double h = std::sqrt(1 - x*x);
        init += h*dx;
      }
      return init;
    },
    /* reduction */
    [](double x, double y) -> double {
      return x + y;
    }
  );
  double pi = 4 * sum;
  return pi;
}

#include <iostream>
#include <limits>

static void warmupTBB();

int main() {
  const int num_intervals = std::numeric_limits<int>::max();

  tbb::tick_count t0 = tbb::tick_count::now();
  double serial_pi = serialPI(num_intervals);
  double serial_time = (tbb::tick_count::now() - t0).seconds();

  warmupTBB();
  tbb::tick_count t1 = tbb::tick_count::now();
  double tbb_pi = parallelPI(num_intervals);
  double tbb_time = (tbb::tick_count::now() - t1).seconds();

  std::cout << "serial_pi == " << serial_pi << std::endl;
  std::cout << "serial_time == " << serial_time << " seconds" << std::endl;
  std::cout << "tbb_pi == " << tbb_pi << std::endl;
  std::cout << "tbb_time == " << tbb_time << " seconds" << std::endl;
  std::cout << "speedup == " << serial_time / tbb_time << std::endl;
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
