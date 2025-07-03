// Pg 50 
/*
The identity argument is a value that leaves other values unchanged when they are
combined with it using the operation that is being parallelized. It is well known that the
identity element with respect to addition (additive identity) is “0” (since x + 0 = x) and
that the identity element with respect to multiplication (multiplicative identity) is “1”
(since x * 1 = x). The reduction function takes two partial results and combines them.
Figure 2-9 shows how func and reduction functions may be applied to compute the
maximum value from an array of 16 elements if the Range is broken into four chunks.
In this example, the associative operation applied by func to the elements of the array
is max() and the identity element is -∞, since max(x,- ∞)=x. In C++, we can use
std::max as the operation and std::numeric_limits<int>::min() as the programmatic
representation of -∞.
*/
#include <limits>
#include <tbb/tbb.h>

int simpleParallelMax(const std::vector<int>& v) {
  int max_value = tbb::parallel_reduce(
    /* the range = */ tbb::blocked_range<int>(0, v.size()), 
    /* identity = */ std::numeric_limits<int>::min(),
    /* func = */ 
    [&](const tbb::blocked_range<int>& r, int init) -> int {
      for (int i = r.begin(); i != r.end(); ++i) {
        init = std::max(init, v[i]);
      }
      return init;
    },
    /* reduction = */ 
    [](int x, int y) -> int {
      return std::max(x,y);
    }
  );
  return max_value;
}

#include <iostream>
int simpleSerialMax(const std::vector<int>&);
static void warmupTBB();

int main() {
  const int N = INT_MAX;
  std::vector<int> a(N, 0);

  // the max value is N/2
  for (int i = 0; i < N; ++i) {
    a[i] = std::abs(i - N/2);
  }

  double serial_time = 0.0, parallel_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    int max_value = simpleSerialMax(a);
    serial_time = (tbb::tick_count::now() - t0).seconds();
    if (max_value != N/2) {
      std::cerr << "ERROR: serial max_value is wrong!" << std::endl;
    }
  }

  warmupTBB();
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    int max_value = simpleParallelMax(a);
    parallel_time = (tbb::tick_count::now() - t0).seconds();
    if (max_value != N/2) {
      std::cerr << "ERROR: parallel max_value is wrong!" << std::endl;
    }
  }

  std::cout << "serial_time == " << serial_time << " seconds" << std::endl
            << "parallel_time == " << parallel_time << " seconds" << std::endl
            << "speedup == " << serial_time/parallel_time << std::endl;

  return 0;
}

int simpleSerialMax(const std::vector<int>& v) {
  int max_value = std::numeric_limits<int>::min();
  for (int i = 0; i < v.size(); ++i) {
    max_value = std::max(max_value,v[i]);
  }
  return max_value;
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
