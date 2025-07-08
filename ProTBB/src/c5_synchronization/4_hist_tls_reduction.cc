// Pg 165
/*
prateek@prateek:~/Workspace/IntelTbb/build/src/c5_synchronization$ ./4_hist_tls_reduction 
Serial: 1.98914, Parallel: 0.186054, Speed-up: 10.6912

10X Speedup !!! 🚅 

🧵 Summary: Using enumerable_thread_specific for Parallel Histogram
- A thread-local object priv_h of type vector<int> is created using enumerable_thread_specific, sized to num_bins.
- Inside a parallel_for, threads process chunks of data.
- Each thread calls priv_h.local() to access its own private histogram:
  - If it’s the first call, a new vector<int> is created.
  - If already called before, the thread reuses its existing private vector.
- This local histogram my_hist is updated within the thread’s chunk.
- Threads handling multiple chunks reuse their private histogram efficiently.

📊 Final Reduction Strategy
- After parallel_for completes, multiple private histograms exist (one per thread).
- The final global histogram hist_p is computed by iterating over all thread-local vectors.
- enumerable_thread_specific supports iteration just like STL containers.
- A loop traverses each thread’s histogram and merges the bin counts into hist_p.

This method leverages thread-local storage for fast, contention-free updates, and combines results efficiently in a 
post-processing step. Elegant and scalable — especially useful in fine-grained parallel histogramming.
*/

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <atomic>
#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

int main(int argc, char** argv) {

  long int n = 1000000000;
  constexpr int num_bins = 256;

  // Initialize random number generator
  std::random_device seed;    // Random device seed
  std::mt19937 mte{seed()};   // mersenne_twister_engine
  std::uniform_int_distribution<> uniform{0,num_bins};
  // Initialize image  
  std::vector<uint8_t> image; // empty vector
  image.reserve(n);           // image vector prealocated
  std::generate_n(std::back_inserter(image), n,
                    [&] { return uniform(mte); }
                 );
  // Initialize histogram
  std::vector<int> hist(num_bins);

  // Serial execution
  tbb::tick_count t0 = tbb::tick_count::now();
  std::for_each(image.begin(), image.end(),
                [&](uint8_t i){hist[i]++;});
  tbb::tick_count t1 = tbb::tick_count::now();
  double t_serial = (t1 - t0).seconds();

  // Parallel execution
  using vector_t = std::vector<int>;
  using priv_h_t = tbb::enumerable_thread_specific<vector_t>;
  priv_h_t priv_h{num_bins};
  t0 = tbb::tick_count::now();
  parallel_for(tbb::blocked_range<size_t>{0, image.size()},
              [&](const tbb::blocked_range<size_t>& r)
              {
                priv_h_t::reference my_hist = priv_h.local();
                for (size_t i = r.begin(); i < r.end(); ++i)
                  my_hist[image[i]]++;
              });
  //Sequential reduction of the private histograms
  vector_t hist_p(num_bins);
  for(auto i=priv_h.begin(); i!=priv_h.end(); ++i){
    for (int j=0; j<num_bins; ++j) hist_p[j]+=(*i)[j];
  }
  t1 = tbb::tick_count::now();
  double t_parallel = (t1 - t0).seconds();

  std::cout << "Serial: "   << t_serial   << ", ";
  std::cout << "Parallel: " << t_parallel << ", ";
  std::cout << "Speed-up: " << t_serial/t_parallel << std::endl;

  if (hist != hist_p)
      std::cerr << "Parallel computation failed!!" << std::endl;

  return 0;
}