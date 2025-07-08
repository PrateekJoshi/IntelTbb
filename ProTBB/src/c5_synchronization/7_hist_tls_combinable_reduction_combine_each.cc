// Pg 169
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <atomic>
#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/combinable.h>

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
  tbb::combinable<vector_t> priv_h{[](){return vector_t(num_bins);}};
  t0 = tbb::tick_count::now();
  parallel_for(tbb::blocked_range<size_t>{0, image.size()},
              [&](const tbb::blocked_range<size_t>& r)
              {
                vector_t& my_hist = priv_h.local();
                for (size_t i = r.begin(); i < r.end(); ++i)
                  my_hist[image[i]]++;
              });
  //Sequential reduction of the private histograms
  vector_t hist_p(num_bins);
  priv_h.combine_each([&](const vector_t& a)
    { // for each priv histogram a
      std::transform(hist_p.begin(),    // source 1 begin
                     hist_p.end(),      // source 1 end
                     a.begin(),         // source 2 begin
                     hist_p.begin(),    // destination begin
                     std::plus<int>() );// binary operation
    });
  t1 = tbb::tick_count::now();
  double t_parallel = (t1 - t0).seconds();

  std::cout << "Serial: "   << t_serial   << ", ";
  std::cout << "Parallel: " << t_parallel << ", ";
  std::cout << "Speed-up: " << t_serial/t_parallel << std::endl;

  if (hist != hist_p)
      std::cerr << "Parallel computation failed!!" << std::endl;

  return 0;
}